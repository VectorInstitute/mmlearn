"""A Module for linear evaluation of pretrained encoders."""

import inspect
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import hydra
import lightning as L  # noqa: N812
import torch
from hydra_zen import store
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall

from mmlearn.datasets.core import Modalities
from mmlearn.modules.layers import MLP


def extract_vision_encoder(
    encoder: Any,
    model_checkpoint_path: Optional[str],
    keys_to_remove: Optional[List[str]] = None,
    keys_to_rename: Optional[Dict[str, str]] = None,  # Default for renaming
    keys_to_ignore: Optional[List[str]] = None,
) -> nn.Module:
    """
    Extract the vision encoder from a PyTorch Lightning model.

    Args:
        encoder (Any): The encoder configuration or model to be instantiated.
        model_checkpoint_path (Optional[str]): Path to the checkpoint file containing
        the encoder's state_dict.
        keys_to_remove (Optional[list]): List of keys to be removed from the state_dict.
        keys_to_rename (Optional[dict]): Dictionary of prefixes or key replacements
        mapping
            old prefixes to new replacements (default removes 'encoders.rgb.').
        keys_to_ignore (Optional[list]): List of keys to ignore when loading the
        state_dict.

    Returns
    -------
        nn.Module: The vision encoder module extracted and initialized.
    """
    model: L.LightningModule = hydra.utils.instantiate(encoder)
    if model_checkpoint_path is None:
        rank_zero_warn(
            "No model_checkpoint_path path was provided for linear evaluation."
        )
    else:
        checkpoint = torch.load(model_checkpoint_path)
        if "state_dict" not in checkpoint:
            raise KeyError("'state_dict' not found in checkpoint")

        state_dict = checkpoint["state_dict"]

        # Remove unwanted keys
        if keys_to_remove:
            state_dict = {
                k: v for k, v in state_dict.items() if k not in keys_to_remove
            }

        # Ignore specific keys
        if keys_to_ignore:
            state_dict = {
                k: v for k, v in state_dict.items() if k not in keys_to_ignore
            }

        # Rename keys based on input mappings
        if keys_to_rename:
            state_dict = {
                k.replace(old_prefix, new_prefix): v
                for k, v in state_dict.items()
                for old_prefix, new_prefix in keys_to_rename.items()
                if k.startswith(old_prefix)
            }

        try:
            if state_dict:
                model["rgb"].load_state_dict(state_dict, strict=True)
                print("Encoder state dict loaded successfully")
        except Exception as e:
            print(f"Error loading state dict: {e}")
    return model["rgb"]


@store(group="task", provider="mmlearn")
class LinearClassifierModule(L.LightningModule):
    """A linear classifier module for evaluating pretrained encoders.

    Parameters
    ----------
    encoder : torch.nn.Module
        A pretrained encoder model, outputting features for the linear classifier.
    modality : str
        The modality of the input data to be passed through the encoder. See
        `common.constants.Modality` for valid values. The target label key is
        inferred from this modality. This means that, for example, that if the
        modality is 'rgb', the target label key is expected to be 'rgb_target'.
    num_output_features : int
        Output features from the encoder, defining the linear classifier's input size.
    num_classes : int
        Number of classes for the classification task.
    hidden_dims : list[int]
        Size of each hidden layer of the model
    task : str
        Classification task type. One of 'binary', 'multiclass', or 'multilabel'.
    freeze_encoder : bool, default = True
        If True, encoder's parameters are frozen during training.
    pre_classifier_batch_norm : bool, default = False
        If True, a batch normalization layer without affine transformation is
        added before the linear classifier, following [1].
    top_k_list : List[int], optional, default = None
        A list of integers specifying the `k` values for top-k accuracy metrics.
        For each `k` in this list, top-k accuracy is calculated and tracked during
        training and validation. This allows for the evaluation of the model's
        performance at predicting the top `k` most probable classes.
    optimizer : DictConfig, optional, default = None
        The configuration for the optimizer. This will be instantiated using
        `hydra.utils.instantiate`, so it should include the `_target_` field,
        which should point to the optimizer class.
    lr_scheduler : DictConfig, optional, default = None
        The configuration for the learning rate scheduler. Two fields are expected:
        `scheduler` (required) and `extras` (optional). The `scheduler` field should
        contain configurations for the learning rate scheduler and must include the
        `_target_` field, which, like the optimizer, should point to the scheduler
        class. The `extras` field may contain one or more of the following:
        - `interval` (str): The interval to apply the learning rate scheduler.
           One of "epoch" or "step". Default is "epoch".
        - `frequency` (int): The frequency to apply the learning rate scheduler
          in the specified interval. Default is 1.
        - `monitor` (str): The metric to monitor for schedulers like ReduceLROnPlateau.
        - `strict` (bool): Whether to strictly enforce the availability of the
          monitored metric (if `True`) or raise a warning if the metric is not found
          (if `False`). Default is `True`.

    Attributes
    ----------
    accuracy_metrics : torchmetrics.MetricCollection
        A collection of metrics that includes accuracy for each `k` in `top_k_list`,
        providing a comprehensive evaluation of model performance across different
        levels of top-k predictions.
    linear_eval : torch.nn.Linear
        Linear classification layer atop the encoder. Input and output features are
        determined by `encoder_output_features` and `num_classes`, respectively.

    References
    ----------
    [1] He, K., Chen, X., Xie, S., Li, Y., Doll'ar, P., & Girshick, R.B. (2021).
        Masked Autoencoders Are Scalable Vision Learners. 2022 IEEE/CVF Conference
        on Computer Vision and Pattern Recognition (CVPR), 15979-15988.
    """

    def __init__(
        self,
        # encoder: torch.nn.Module,
        encoder: nn.Module,
        model_checkpoint_path: Optional[str],  # change name
        modality: str,
        num_output_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        freeze_encoder: bool = True,
        pre_classifier_batch_norm: bool = False,
        top_k_list: Optional[List[int]] = None,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                Dict[str, partial[torch.optim.lr_scheduler.LRScheduler]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
    ):
        super().__init__()
        assert task in ["binary", "multiclass", "multilabel"], (
            f"Invalid task type: {task}. "
            "Expected one of ['binary', 'multiclass', 'multilabel']."
        )

        self.modality = modality

        self.encoder: nn.Module = extract_vision_encoder(
            encoder, model_checkpoint_path, keys_to_rename={"encoders.rgb.": ""}
        )

        linear_layer = MLP(num_output_features, num_classes, hidden_dims)

        if pre_classifier_batch_norm:
            linear_layer = nn.Sequential(
                nn.BatchNorm1d(num_output_features, affine=False),
                linear_layer,
            )
        self.classifier = linear_layer

        self.freeze_encoder = freeze_encoder
        self.num_classes = num_classes

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

        self.top_k_list = top_k_list
        if task == "multiclass":
            if self.top_k_list is None:
                self.top_k_list = [1, 5]
            accuracy_metrics = {
                f"top_{k}_accuracy": Accuracy(
                    task=task, num_classes=num_classes, top_k=k
                )
                for k in self.top_k_list
            }

            # Additional metrics for multiclass classification
            additional_metrics = {
                "precision": Precision(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "recall": Recall(task=task, num_classes=num_classes, average="macro"),
                "f1_score": F1Score(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "auc": AUROC(
                    task=task, num_classes=num_classes, average="macro"
                ),  # AUROC for multiclass
            }

        elif task == "multilabel":
            # Accuracy and other metrics for multilabel classification
            accuracy_metrics = {"accuracy": Accuracy(task=task, num_labels=num_classes)}

            # Additional metrics for multilabel classification
            additional_metrics = {
                "precision": Precision(
                    task=task, num_labels=num_classes, average="macro"
                ),
                "recall": Recall(task=task, num_labels=num_classes, average="macro"),
                "f1_score": F1Score(task=task, num_labels=num_classes, average="macro"),
                "auc": AUROC(task=task, num_labels=num_classes),  # AUC for multilabel
            }

        else:  # binary
            # Accuracy and other metrics for binary classification
            accuracy_metrics = {"accuracy": Accuracy(task=task)}

            # Additional metrics for binary classification
            additional_metrics = {
                "precision": Precision(task=task),
                "recall": Recall(task=task),
                "f1_score": F1Score(task=task),
                "auc": AUROC(task=task),  # AUROC for binary classification
            }

        # combine all metrics
        metrics = MetricCollection({**accuracy_metrics, **additional_metrics})
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the encoder and linear classifier.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The logits predicted by the classifier.
        """
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            x = self.encoder(x)
        return self.classifier(x[0])

    def _get_logits_and_labels(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and labels for a batch of data."""
        x: torch.Tensor = batch
        y = batch[Modalities.get_modality(self.modality).target]

        logits = self(x)
        return logits, y

    def _compute_loss(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if self.loss_fn is None:
            return None

        if self.freeze_encoder:
            self.encoder.eval()

        logits, y = self._get_logits_and_labels(batch)

        loss: torch.Tensor = self.loss_fn(logits, y)
        self.train_metrics.update(logits, y)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        loss = self._compute_loss(batch)

        if loss is None:
            raise ValueError("The loss function must be provided for training.")

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Execute a validation step using a single batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The current batch of validation data, including input tensors and labels.
        batch_idx : int
            The index of the current validation batch.

        Returns
        -------
        torch.Tensor
            The loss computed for the batch.
        """
        logits, y = self._get_logits_and_labels(batch)

        loss: torch.Tensor = self.loss_fn(logits, y)
        self.log("val/loss", self.all_gather(loss.clone().detach()).mean())

        self.valid_metrics.update(logits, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics accumulated over the epoch."""
        val_metrics = self.valid_metrics.compute()
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value.item()}")
        self.log_dict(val_metrics)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:  # noqa: PLR0912
        """Configure the optimizer and learning rate scheduler."""
        if self.optimizer is None:
            rank_zero_warn(
                "Optimizer not provided. Training will continue without an optimizer. "
                "LR scheduler will not be used.",
            )
            return None

        weight_decay: Optional[float] = self.optimizer.keywords.get(
            "weight_decay", None
        )
        if weight_decay is None:  # try getting default value
            kw_param = inspect.signature(self.optimizer.func).parameters.get(
                "weight_decay"
            )
            if kw_param is not None and kw_param.default != inspect.Parameter.empty:
                weight_decay = kw_param.default

        parameters = [param for param in self.parameters() if param.requires_grad]

        if weight_decay is not None:
            decay_params = []
            no_decay_params = []

            for param in self.parameters():
                if not param.requires_grad:
                    continue

                if param.ndim < 2:  # includes all bias and normalization parameters
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            parameters = [
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "name": "weight_decay_params",
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "name": "no_weight_decay_params",
                },
            ]

        optimizer = self.optimizer(parameters)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "Expected optimizer to be an instance of `torch.optim.Optimizer`, "
                f"but got {type(optimizer)}.",
            )

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                if "scheduler" not in self.lr_scheduler:
                    raise ValueError(
                        "Expected 'scheduler' key in the learning rate scheduler dictionary.",
                    )

                lr_scheduler = self.lr_scheduler["scheduler"](optimizer)
                if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                    raise TypeError(
                        "Expected scheduler to be an instance of `torch.optim.lr_scheduler.LRScheduler`, "
                        f"but got {type(lr_scheduler)}.",
                    )
                lr_scheduler_dict: Dict[
                    str, Union[torch.optim.lr_scheduler.LRScheduler, Any]
                ] = {"scheduler": lr_scheduler}

                if self.lr_scheduler.get("extras"):
                    extras = self.lr_scheduler["extras"]
                    if isinstance(extras, partial):
                        # Extract the keywords from the partial object
                        lr_scheduler_dict.update(extras.keywords)

                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

            lr_scheduler = self.lr_scheduler(optimizer)
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise TypeError(
                    "Expected scheduler to be an instance of `torch.optim.lr_scheduler.LRScheduler`, "
                    f"but got {type(lr_scheduler)}.",
                )
            return [optimizer], [lr_scheduler]

        return optimizer
