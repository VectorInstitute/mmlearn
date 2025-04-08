"""A Module for linear evaluation of pretrained encoders."""

import re
import warnings
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import torch
from hydra_zen import store
from torch import nn
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall

from mmlearn.datasets.core import Modalities
from mmlearn.modules.layers import MLP
from mmlearn.tasks.base import TrainingTask


@store(group="task", provider="mmlearn")
class LinearEvaluation(TrainingTask):
    """Linear evaluation task for pretrained encoders.

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
    top_k_list : list[int], optional, default = None
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
        encoder: nn.Module,
        checkpoint_path: Optional[str],
        modality: str,
        num_output_features: int,
        num_classes: int,
        pre_classifier_batch_norm: bool = False,
        classifier_hidden_dims: Optional[list[int]] = None,
        classifier_norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        classifier_activation_layer: Optional[
            Callable[..., torch.nn.Module]
        ] = torch.nn.ReLU,
        classifier_bias: Union[bool, list[bool]] = True,
        classifier_dropout: Union[float, list[float]] = 0.0,
        freeze_encoder: bool = True,
        encoder_input_kwargs: Optional[dict[str, Any]] = None,
        encoder_outputs_processor: Optional[Callable[..., torch.Tensor]] = None,
        encoder_state_dict_key: str = "state_dict",
        state_dict_pattern_replacement_map: Optional[dict[str, str]] = None,
        state_dict_patterns_to_exclude: Optional[list[str]] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        top_k_list: Optional[list[int]] = None,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                dict[str, partial[torch.optim.lr_scheduler.LRScheduler]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
    ) -> None:
        # input validation
        assert task in ["binary", "multiclass", "multilabel"], (
            f"Invalid task type: {task}. "
            "Expected one of ['binary', 'multiclass', 'multilabel']."
        )

        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=nn.CrossEntropyLoss()
            if task == "multiclass"
            else nn.BCEWithLogitsLoss(),
            compute_validation_loss=compute_validation_loss,
            compute_test_loss=compute_test_loss,
        )

        self.encoder = encoder
        self.modality = Modalities.get_modality(modality)
        self.num_output_features = num_output_features
        self.num_classes = num_classes
        self.pre_classifier_batch_norm = pre_classifier_batch_norm
        self.freeze_encoder = freeze_encoder
        self.encoder_outputs_processor = encoder_outputs_processor
        self.task = task
        self.top_k_list = top_k_list
        self.encoder_input_kwargs = encoder_input_kwargs

        checkpoint_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        state_dict = get_state_dict(
            checkpoint_dict,
            state_dict_key=encoder_state_dict_key,
            pattern_replacement_map=state_dict_pattern_replacement_map,
            patterns_to_exclude=state_dict_patterns_to_exclude,
        )
        self.encoder.load_state_dict(state_dict)

        linear_layer = MLP(
            in_dim=num_output_features,
            out_dim=num_classes,
            hidden_dims=classifier_hidden_dims,
            norm_layer=classifier_norm_layer,
            activation_layer=classifier_activation_layer,
            bias=classifier_bias,
            dropout=classifier_dropout,
        )

        if pre_classifier_batch_norm:
            linear_layer = nn.Sequential(
                nn.BatchNorm1d(num_output_features, affine=False),
                linear_layer,
            )
        self.classifier = linear_layer

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if task == "multiclass":
            if self.top_k_list is None:
                self.top_k_list = [1, 5]

            metrics = MetricCollection(
                {
                    f"top_{k}_accuracy": Accuracy(
                        task=task, num_classes=num_classes, top_k=k
                    )
                    for k in self.top_k_list
                }
            )

            metrics.add_metrics(
                {
                    "precision": Precision(
                        task=task, num_classes=num_classes, average="macro"
                    ),
                    "recall": Recall(
                        task=task, num_classes=num_classes, average="macro"
                    ),
                    "f1_score": F1Score(
                        task=task, num_classes=num_classes, average="macro"
                    ),
                    "auc": AUROC(task=task, num_classes=num_classes, average="macro"),
                }
            )
        elif task == "multilabel":
            metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task=task, num_labels=num_classes),
                    "precision": Precision(
                        task=task, num_labels=num_classes, average="macro"
                    ),
                    "recall": Recall(
                        task=task, num_labels=num_classes, average="macro"
                    ),
                    "f1_score": F1Score(
                        task=task, num_labels=num_classes, average="macro"
                    ),
                    "auc": AUROC(task=task, num_labels=num_classes),
                }
            )
        else:  # binary
            metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task=task),
                    "precision": Precision(task=task),
                    "recall": Recall(task=task),
                    "f1_score": F1Score(task=task),
                    "auc": AUROC(task=task),
                }
            )

        self._metrics = {
            "train": metrics.clone(prefix="train/"),
            "val": metrics.clone(prefix="val/"),
            "test": metrics.clone(prefix="test/"),
        }

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the encoder and linear classifier.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Dictionary containing input tensors for the encoder.

        Returns
        -------
        torch.Tensor
            The logits predicted by the classifier.
        """
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            enc_out = self.encoder(inputs, **self.encoder_input_kwargs)
            if self.encoder_outputs_processor is not None:
                enc_out = self.encoder_outputs_processor(enc_out)

        return self.classifier(enc_out)

    def on_fit_start(self) -> None:
        """Move the metrics to the device of the Lightning module."""
        self._metrics = {
            step_name: metric.to(self.device)
            for step_name, metric in self._metrics.items()
            if step_name in ["train", "val"]
        }

    def on_train_epoch_start(self) -> None:
        """Set the encoder to evaluation mode if it is frozen."""
        self.encoder = self.encoder.train(mode=not self.freeze_encoder)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : dict[str, Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        """Compute metrics at the end of a training epoch."""
        self._on_epoch_end("train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute a validation step using a single batch.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The current batch of validation data, including input tensors and labels.
        batch_idx : int
            The index of the current validation batch.

        Returns
        -------
        torch.Tensor
            The loss computed for the batch.
        """
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics accumulated over the epoch."""
        self._on_epoch_end("val")

    def on_test_start(self) -> None:
        """Move the metrics to the device of the Lightning module."""
        self._metrics["test"] = self._metrics["test"].to(self.device)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute a test step using a single batch.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The current batch of test data, including input tensors and labels.
        batch_idx : int
            The index of the current test batch.

        Returns
        -------
        torch.Tensor
            The loss computed for the batch.
        """
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Compute test metrics accumulated over the epoch."""
        self._on_epoch_end("test")

    def _shared_step(
        self, batch: dict[str, torch.Tensor], step_name: Literal["train", "val", "test"]
    ) -> Optional[torch.Tensor]:
        """
        Execute a shared step for training, validation, or testing.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The current batch of data.
        step_name : Literal["train", "val", "test"]
            The name of the step to execute.
        """
        if step_name == "train" and self.loss_fn is None:
            raise ValueError("The loss function must be provided for training.")

        logits = self(batch)
        y = batch[self.modality.target]

        if self.loss_fn is not None:
            loss: torch.Tensor = self.loss_fn(logits, y)
            self.log(f"{step_name}/loss", loss, prog_bar=True, sync_dist=True)

        self._metrics[step_name].update(logits, y)

        return loss if self.loss_fn is not None else None

    def _on_epoch_end(self, step_name: Literal["train", "val", "test"]) -> None:
        """
        Compute metrics at the end of an epoch.

        Parameters
        ----------
        step_name : Literal["train", "val", "test"]
            The name of the step to execute
        """
        metrics = self._metrics[step_name].compute()
        self.log_dict(metrics, prog_bar=step_name in ["val", "test"])
        self._metrics[step_name].reset()


def get_state_dict(  # noqa: PLR0912
    checkpoint_dict: dict[str, Any],
    state_dict_key: str = "state_dict",
    pattern_replacement_map: Optional[dict[str, str]] = None,
    patterns_to_exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Process a state dictionary by applying regex pattern replacements and exclusions.

    Parameters
    ----------
    checkpoint_dict : dict[str, Any]
        Dictionary containing the state dict in one of its keys.
    state_dict_key : str, default="state_dict"
        Key in ``checkpoint_dict`` containing the state dictionary to process.
    pattern_replacement_map : dict[str, str], optional, default=None
        Dictionary mapping regex patterns to their replacement strings.
    patterns_to_exclude : list[str], optional, default=None
        List of regex patterns for keys to exclude from the processed state dictionary.

    Returns
    -------
        Processed state dictionary

    Raises
    ------
    TypeError
        If inputs are not of expected types.
    KeyError
        If state_dict_key is not in ``checkpoint_dict``.
    ValueError
        If regex patterns are invalid.
    """
    if not isinstance(checkpoint_dict, dict):
        raise TypeError(
            "Expected ``checkpoint_dict`` to be a dictionary, "
            f"but got {type(checkpoint_dict)}"
        )
    if state_dict_key not in checkpoint_dict:
        raise KeyError(
            f"Key '{state_dict_key}' not found in ``checkpoint_dict``. "
            f"Available keys: {list(checkpoint_dict.keys())}"
        )

    state_dict = checkpoint_dict[state_dict_key]
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Expected state dictionary in ``checkpoint_dict`` to be a dictionary, "
            f"but got {type(state_dict)}"
        )

    if pattern_replacement_map is None:
        pattern_replacement_map = {}
    if patterns_to_exclude is None:
        patterns_to_exclude = []

    if not isinstance(pattern_replacement_map, dict):
        raise TypeError(
            "Expected ``pattern_replacement_map`` to be a dictionary, "
            f"but got {type(pattern_replacement_map)}"
        )
    if not isinstance(patterns_to_exclude, list):
        raise TypeError(
            "Expected ``patterns_to_exclude`` to be a list, "
            f"but got {type(patterns_to_exclude)}"
        )

    processed_state_dict = {}

    # apply pattern replacements
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Dictionary keys must be strings for regex operations, found {type(key)}"
            )

        new_key = key
        for pattern, replacement in pattern_replacement_map.items():
            try:
                new_key = re.sub(pattern, replacement, new_key)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {str(e)}") from e

        # check for key collisions
        if new_key in processed_state_dict:
            warnings.warn(
                f"Key '{new_key}' already exists and will be overwritten.",
                UserWarning,
                stacklevel=2,
            )

        processed_state_dict[new_key] = value

    # apply exclusions
    if patterns_to_exclude:
        filtered_dict = {}
        for key, value in processed_state_dict.items():
            exclude_key = False
            for pattern in patterns_to_exclude:
                try:
                    if re.match(pattern, key):
                        exclude_key = True
                        break
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex pattern '{pattern}' in exclusion list: {str(e)}"
                    ) from e

            if not exclude_key:
                filtered_dict[key] = value

        processed_state_dict = filtered_dict

    return processed_state_dict


@store(group="helpers", provider="mmlearn", zen_partial=False)  # type: ignore[misc]
def avg_pool_last_n_hidden_states(
    encoder_output: tuple[torch.Tensor, Optional[list[torch.Tensor]]], n: int = 1
) -> torch.Tensor:
    """Average pool the last ``n`` intermediate layer outputs of an encoder.

    Parameters
    ----------
    encoder_output : tuple[torch.Tensor, Optional[list[torch.Tensor]]]
        Tuple of encoder outputs where the first element is the output of the last layer
        and the second element is an optional list of intermediate layer outputs.
    n : int, default=1
        The number of layers to average pool.

    Returns
    -------
    torch.Tensor
        The average pooled encoder output.

    Raises
    ------
    ValueError
        If intermediate layer outputs are not available or if ``n`` is less than 1
        or greater than the number of available intermediate layers.
    """
    if encoder_output[1] is None:
        raise ValueError("Intermediate layer outputs are not available.")

    if n < 1:
        raise ValueError("Number of layers to average pool must be greater than 0.")

    if n > len(encoder_output[1]):
        raise ValueError(
            f"Requested {n} layers for average pooling, but only {len(encoder_output[1])} "
            "intermediate layers are available."
        )
    # each layer output is a tensor of shape (batch_size, num_patches, num_features)
    # take the average across the num_patches dimension, then concatenate the results
    return torch.cat(
        [layer_output.mean(dim=1) for layer_output in encoder_output[1][-n:]],
        dim=-1,
    )
