"""Contrastive pretraining task."""

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import lightning as L  # noqa: N812
import torch
import torch.distributed
import torch.distributed.nn
from hydra_zen import store
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn

from mmlearn.datasets.core import Modalities, find_matching_indices
from mmlearn.datasets.core.modalities import Modality
from mmlearn.modules.losses import CLIPLoss
from mmlearn.tasks.hooks import EvaluationHooks


_unsupported_modality_error = (
    "Found unsupported modality `{}` in the input. Supported modalities are "
    f"{Modalities.list_modalities()}."
    "HINT: New modalities can be added with `Modalities.register_modality` method."
)


@dataclass
class ModuleKeySpec:
    """Module key specification for mapping modules to modalities."""

    encoder_key: Optional[str] = None
    head_key: Optional[str] = None
    postprocessor_key: Optional[str] = None


@dataclass
class LossPairSpec:
    """Specification for a pair of modalities to compute the contrastive loss."""

    modalities: Tuple[str, str]
    weight: float = 1.0


@dataclass
class AuxiliaryTaskSpec:
    """Specification for an auxiliary task to run alongside the main task."""

    modality: str
    task: Any  # `functools.partial[L.LightningModule]` expected
    loss_weight: float = 1.0


@dataclass
class EvaluationSpec:
    """Specification for an evaluation task."""

    task: Any  # `EvaluationHooks` expected
    run_on_validation: bool = True
    run_on_test: bool = True


@store(group="task", provider="mmlearn")
class ContrastivePretraining(L.LightningModule):
    """Contrastive pretraining task.

    This class supports contrastive pretraining with `N` modalities of data. It
    allows the sharing of encoders, heads, and postprocessors across modalities.
    It also supports computing the contrastive loss between specified pairs of
    modalities, as well as training auxiliary tasks alongside the main contrastive
    pretraining task.

    Parameters
    ----------
    encoders : Dict[str, nn.Module]
        A dictionary of encoders. The keys can be any string, including the names of
        any supported modalities. If the keys are not supported modalities, the
        `modality_module_mapping` parameter must be provided to map the encoders to
        specific modalities. The encoders are expected to take a dictionary of input
        values and return a list-like object with the first element being the encoded
        values. This first element is passed on to the heads or postprocessors and
        the remaining elements are ignored.
    heads : Dict[str, Union[nn.Module, Dict[str, nn.Module]]], optional, default=None
        A dictionary of modules that process the encoder outputs, usually projection
        heads. If the keys do not correspond to the name of a supported modality,
        the `modality_module_mapping` parameter must be provided. If any of the values
        are dictionaries, they will be wrapped in a `nn.Sequential` module. All
        head modules are expected to take a single input tensor and return a single
        output tensor.
    postprocessors : Dict[str, Union[nn.Module, Dict[str, nn.Module]]], optional, default=None
        A dictionary of modules that process the head outputs. If the keys do not
        correspond to the name of a supported modality, the `modality_module_mapping`
        parameter must be provided. If any of the values are dictionaries, they will
        be wrapped in a `nn.Sequential` module. All postprocessor modules are expected
        to take a single input tensor and return a single output tensor.
    modality_module_mapping : Dict[str, ModuleKeySpec], optional, default=None
        A dictionary mapping modalities to encoders, heads, and postprocessors.
        Useful for reusing the same instance of a module across multiple modalities.
    optimizer : partial[torch.optim.Optimizer], optional, default=None
        The optimizer to use for training. This is expected to be a partial function,
        created using `functools.partial`, that takes the model parameters as the
        only required argument. If not provided, training will continue without an
        optimizer.
    lr_scheduler : Union[Dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]], partial[torch.optim.lr_scheduler.LRScheduler]], optional, default=None
        The learning rate scheduler to use for training. This can be a partial function
        that takes the optimizer as the only required argument or a dictionary with
        a `scheduler` key that specifies the scheduler and an optional `extras` key
        that specifies additional arguments to pass to the scheduler. If not provided,
        the learning rate will not be adjusted during training.
    loss : CLIPLoss, optional, default=None
        The loss function to use.
    modality_loss_pairs : List[LossPairSpec], optional, default=None
        A list of pairs of modalities to compute the contrastive loss between and
        the weight to apply to each pair.
    auxiliary_tasks : Dict[str, AuxiliaryTaskSpec], optional, default=None
        Auxiliary tasks to run alongside the main contrastive pretraining task.
        The auxiliary task module is expected to be a partially-initialized instance
        of a `LightningModule` created using `functools.partial`, such that an
        initialized encoder can be passed as the only argument. The `modality`
        parameter specifies the modality of the encoder to use for the auxiliary task.
        The `loss_weight` parameter specifies the weight to apply to the auxiliary
        task loss.
    log_auxiliary_tasks_loss : bool, optional, default=False
        Whether to log the loss of auxiliary tasks to the main logger.
    compute_validation_loss : bool, optional, default=True
        Whether to compute the validation loss if a validation dataloader is provided.
        The loss function must be provided to compute the validation loss.
    compute_test_loss : bool, optional, default=True
        Whether to compute the test loss if a test dataloader is provided. The loss
        function must be provided to compute the test loss.
    evaluation_tasks : Dict[str, EvaluationSpec], optional, default=None
        Evaluation tasks to run during validation, while training, and during testing.

    """  # noqa: W505

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        encoders: Dict[str, nn.Module],
        heads: Optional[Dict[str, Union[nn.Module, Dict[str, nn.Module]]]] = None,
        postprocessors: Optional[
            Dict[str, Union[nn.Module, Dict[str, nn.Module]]]
        ] = None,
        modality_module_mapping: Optional[Dict[str, ModuleKeySpec]] = None,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                Dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        loss: Optional[CLIPLoss] = None,
        modality_loss_pairs: Optional[List[LossPairSpec]] = None,
        auxiliary_tasks: Optional[Dict[str, AuxiliaryTaskSpec]] = None,
        log_auxiliary_tasks_loss: bool = False,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[Dict[str, EvaluationSpec]] = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        if modality_module_mapping is None:
            # assume all the module dictionaries use the same keys corresponding
            # to modalities
            modality_module_mapping = {}
            for key in encoders:
                modality_module_mapping[key] = ModuleKeySpec(
                    encoder_key=key,
                    head_key=key,
                    postprocessor_key=key,
                )

        # match modalities to encoders, heads, and postprocessors
        modality_encoder_mapping: Dict[str, Optional[str]] = {}
        modality_head_mapping: Dict[str, Optional[str]] = {}
        modality_postprocessor_mapping: Dict[str, Optional[str]] = {}
        for modality_key, module_mapping in modality_module_mapping.items():
            if not Modalities.has_modality(modality_key):
                raise ValueError(_unsupported_modality_error.format(modality_key))
            modality_encoder_mapping[modality_key] = module_mapping.encoder_key
            modality_head_mapping[modality_key] = module_mapping.head_key
            modality_postprocessor_mapping[modality_key] = (
                module_mapping.postprocessor_key
            )

        # ensure all modules are mapped to a modality
        for key in encoders:
            if key not in modality_encoder_mapping.values():
                if not Modalities.has_modality(key):
                    raise ValueError(_unsupported_modality_error.format(key))
                modality_encoder_mapping[key] = key

        if heads is not None:
            for key in heads:
                if key not in modality_head_mapping.values():
                    if not Modalities.has_modality(key):
                        raise ValueError(_unsupported_modality_error.format(key))
                    modality_head_mapping[key] = key

        if postprocessors is not None:
            for key in postprocessors:
                if key not in modality_postprocessor_mapping.values():
                    if not Modalities.has_modality(key):
                        raise ValueError(_unsupported_modality_error.format(key))
                    modality_postprocessor_mapping[key] = key

        self._available_modalities: List[Modality] = [
            Modalities.get_modality(modality_key)
            for modality_key in modality_encoder_mapping
        ]
        assert (
            len(self._available_modalities) >= 2
        ), "Expected at least two modalities to be available. "

        self.encoders = nn.ModuleDict(
            {
                Modalities.get_modality(modality_key): encoders[encoder_key]
                for modality_key, encoder_key in modality_encoder_mapping.items()
                if encoder_key is not None
            }
        )
        self.heads = None
        if heads is not None:
            self.heads = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key): heads[head_key]
                    if isinstance(heads[head_key], nn.Module)
                    else nn.Sequential(*heads[head_key].values())
                    for modality_key, head_key in modality_head_mapping.items()
                    if head_key is not None
                }
            )

        self.postprocessors = None
        if postprocessors is not None:
            self.postprocessors = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key): postprocessors[
                        postprocessor_key
                    ]
                    if isinstance(postprocessors[postprocessor_key], nn.Module)
                    else nn.Sequential(*postprocessors[postprocessor_key].values())
                    for modality_key, postprocessor_key in modality_postprocessor_mapping.items()
                    if postprocessor_key is not None
                }
            )

        if modality_loss_pairs is None:
            modality_loss_pairs = [
                LossPairSpec(modalities=(m1.value, m2.value))
                for m1, m2 in itertools.combinations(self._available_modalities, 2)
            ]

        for modality_pair in modality_loss_pairs:
            if not all(
                Modalities.get_modality(modality) in self._available_modalities
                for modality in modality_pair.modalities
            ):
                raise ValueError(
                    "Found unspecified modality in the loss pair specification "
                    f"{modality_pair.modalities}. Available modalities are "
                    f"{self._available_modalities}."
                )
        self.modality_loss_pairs = modality_loss_pairs

        self.aux_task_specs = auxiliary_tasks or {}
        self.auxiliary_tasks: Dict[str, L.LightningModule] = {}
        for task_name, task_spec in self.aux_task_specs.items():
            if not Modalities.has_modality(task_spec.modality):
                raise ValueError(
                    f"Found unsupported modality `{task_spec.modality}` in the auxiliary tasks. "
                    f"Available modalities are {self._available_modalities}."
                )
            if not isinstance(task_spec.task, partial):
                raise TypeError(
                    f"Expected auxiliary task to be a partial function, but got {type(task_spec.task)}."
                )

            self.auxiliary_tasks[task_name] = task_spec.task(
                self.encoders[Modalities.get_modality(task_spec.modality)]
            )

        if loss is None and (compute_validation_loss or compute_test_loss):
            raise ValueError(
                "Loss function must be provided to compute validation or test loss."
            )

        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_auxiliary_tasks_loss = log_auxiliary_tasks_loss
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

        if evaluation_tasks is not None:
            for eval_task_spec in evaluation_tasks.values():
                if not isinstance(eval_task_spec.task, EvaluationHooks):
                    raise TypeError(
                        f"Expected {eval_task_spec.task} to be an instance of `EvaluationHooks` "
                        f"but got {type(eval_task_spec.task)}."
                    )
        self.evaluation_tasks = evaluation_tasks

    def encode(
        self, inputs: Dict[Union[str, Modality], Any], modality: Modality
    ) -> torch.Tensor:
        """Encode the input values for the given modality.

        Parameters
        ----------
        inputs : Dict[Union[str, Modality], Any]
            Input values.
        modality : Modality
            The modality to encode.

        Returns
        -------
        torch.Tensor
            The encoded values for the specified modality.
        """
        output = self.encoders[modality](inputs)[0]

        if self.heads and modality in self.heads:
            output = self.heads[modality](output)

        if self.postprocessors and modality in self.postprocessors:
            output = self.postprocessors[modality](output)

        return output

    def forward(
        self, inputs: Dict[Union[str, Modality], Any]
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[Union[str, Modality], Any]
            The input tensors to encode.

        Returns
        -------
        Dict[str, torch.Tensor]
            The encodings for each modality.
        """
        outputs = {
            modality.embedding: self.encode(inputs, modality)
            for modality in self._available_modalities
        }

        if not all(
            output.size(-1) == list(outputs.values())[0].size(-1)
            for output in outputs.values()
        ):
            raise ValueError("Expected all model outputs to have the same dimension.")

        return outputs

    def _compute_loss(
        self,
        batch: Dict[Union[str, Modality], Any],
        batch_idx: int,
        outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.loss_fn is None:
            return None

        contrastive_losses: list[torch.Tensor] = []
        for loss_pair in self.modality_loss_pairs:
            modality_a = Modalities.get_modality(loss_pair.modalities[0])
            modality_b = Modalities.get_modality(loss_pair.modalities[1])

            indices_a, indices_b = find_matching_indices(
                batch["example_ids"][modality_a],
                batch["example_ids"][modality_b],
            )
            if indices_a.numel() == 0 or indices_b.numel() == 0:
                continue

            contrastive_losses.append(
                self.loss_fn(
                    outputs[modality_a.embedding][indices_a],
                    outputs[modality_b.embedding][indices_b],
                )
                * loss_pair.weight
            )

        auxiliary_losses: list[torch.Tensor] = []
        if self.auxiliary_tasks:
            for task_name, task_spec in self.aux_task_specs.items():
                auxiliary_task_output = self.auxiliary_tasks[task_name].training_step(
                    batch, batch_idx
                )
                if isinstance(auxiliary_task_output, torch.Tensor):
                    auxiliary_task_loss = auxiliary_task_output
                elif isinstance(auxiliary_task_output, Mapping):
                    auxiliary_task_loss = auxiliary_task_output["loss"]
                else:
                    raise ValueError(
                        "Expected auxiliary task output to be a tensor or a mapping "
                        f"containing a 'loss' key, but got {type(auxiliary_task_output)}."
                    )

                auxiliary_losses.append(task_spec.loss_weight * auxiliary_task_loss)
                if self.log_auxiliary_tasks_loss:
                    self.log(
                        f"train/{task_name}_loss",
                        auxiliary_task_loss
                        if not self.fabric
                        else self.all_gather(
                            auxiliary_task_loss.clone().detach()
                        ).mean(),
                        sync_dist=True,
                    )

        return torch.stack(contrastive_losses + auxiliary_losses).sum()

    def training_step(
        self, batch: Dict[Union[str, Modality], Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        outputs = self(batch)
        loss = self._compute_loss(batch, batch_idx, outputs)
        if loss is None:
            raise ValueError("The loss function must be provided for training.")

        self.log(
            "train/loss",
            loss if not self.fabric else self.all_gather(loss.clone().detach()).mean(),
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare for the validation epoch."""
        self._on_eval_epoch_start("val")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor or None
            The loss for the batch or None if the loss function is not provided.
        """
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level metrics at the end of the validation epoch."""
        self._on_eval_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        """Prepare for the test epoch."""
        self._on_eval_epoch_start("test")

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single test step.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor or None
            The loss for the batch or None if the loss function is not provided.
        """
        return self._shared_eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self) -> None:
        """Compute and log epoch-level metrics at the end of the test epoch."""
        self._on_eval_epoch_end("test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler."""
        if self.optimizer is None:
            rank_zero_warn(
                "Optimizer not provided. Training will continue without an optimizer. "
                "LR scheduler will not be used.",
            )
            return None
        # TODO: add mechanism to exclude certain parameters from weight decay
        optimizer = self.optimizer(self.parameters())
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
                    lr_scheduler_dict.update(self.lr_scheduler["extras"])
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

            lr_scheduler = self.lr_scheduler(optimizer)
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise TypeError(
                    "Expected scheduler to be an instance of `torch.optim.lr_scheduler.LRScheduler`, "
                    f"but got {type(lr_scheduler)}.",
                )
            return [optimizer], [lr_scheduler]

        return optimizer

    def _on_eval_epoch_start(self, eval_type: Literal["val", "test"]) -> None:
        """Prepare for the evaluation epoch."""
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.on_evaluation_epoch_start(self)

    def _shared_eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        eval_type: Literal["val", "test"],
    ) -> Optional[torch.Tensor]:
        """Run a single evaluation step.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor or None
            The loss for the batch or None if the loss function is not provided.
        """
        outputs = self(batch)
        loss: Optional[torch.Tensor] = None
        if (eval_type == "val" and self.compute_validation_loss) or (
            eval_type == "test" and self.compute_test_loss
        ):
            loss = self._compute_loss(batch, batch_idx, outputs)
            if loss is not None:
                self.log(
                    f"{eval_type}/loss",
                    loss
                    if not self.fabric
                    else self.all_gather(loss.clone().detach()).mean(),
                    prog_bar=True,
                    sync_dist=True,
                )

        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    batch_result = task_spec.task.evaluation_step(
                        self.trainer, self, batch, batch_idx
                    )
                    if batch_result:
                        for key, value in batch_result.items():
                            self.log(
                                f"{eval_type}/{key}_step",
                                value,
                                on_step=True,
                                on_epoch=False,
                                sync_dist=True,
                            )
        return loss

    def _on_eval_epoch_end(self, eval_type: Literal["val", "test"]) -> None:
        """Compute and log epoch-level metrics at the end of the evaluation epoch."""
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    results = task_spec.task.on_evaluation_epoch_end(self)
                    if results:
                        for key, value in results.items():
                            self.log(f"{eval_type}/{key}", value)
