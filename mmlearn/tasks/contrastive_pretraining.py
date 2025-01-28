"""Contrastive pretraining task."""

import itertools
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping, Optional, Union

import lightning as L  # noqa: N812
import numpy as np
import torch
import torch.distributed
import torch.distributed.nn
from hydra_zen import store
from torch import nn

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.tasks.base import TrainingTask
from mmlearn.tasks.hooks import EvaluationHooks


_unsupported_modality_error = (
    "Found unsupported modality `{}` in the input. Supported modalities are "
    f"{Modalities.list_modalities()}."
    "HINT: New modalities can be added with `Modalities.register_modality` method."
)


@dataclass
class ModuleKeySpec:
    """Module key specification for mapping modules to modalities."""

    #: The key of the encoder module. If not provided, the modality name is used.
    encoder_key: Optional[str] = None

    #: The key of the head module. If not provided, the modality name is used.
    head_key: Optional[str] = None

    #: The key of the postprocessor module. If not provided, the modality name is used.
    postprocessor_key: Optional[str] = None


@dataclass
class LossPairSpec:
    """Specification for a pair of modalities to compute the contrastive loss."""

    #: The pair of modalities to compute the contrastive loss between.
    modalities: tuple[str, str]

    #: The weight to apply to the contrastive loss for the pair of modalities.
    weight: float = 1.0


@dataclass
class AuxiliaryTaskSpec:
    """Specification for an auxiliary task to run alongside the main task."""

    #: The modality of the encoder to use for the auxiliary task.
    modality: str

    #: The auxiliary task module. This is expected to be a partially-initialized
    #: instance of a :py:class:`~lightning.pytorch.core.LightningModule` created
    #: using :py:func:`functools.partial`, such that an initialized encoder can be
    #: passed as the only argument.
    task: Any  # `functools.partial[L.LightningModule]` expected

    #: The weight to apply to the auxiliary task loss.
    loss_weight: float = 1.0


@dataclass
class EvaluationSpec:
    """Specification for an evaluation task."""

    #: The evaluation task module. This is expected to be an instance of
    #: :py:class:`~mmlearn.tasks.hooks.EvaluationHooks`.
    task: Any  # `EvaluationHooks` expected

    #: Whether to run the evaluation task during validation.
    run_on_validation: bool = True

    #: Whether to run the evaluation task during training.
    run_on_test: bool = True


@store(group="task", provider="mmlearn")
class ContrastivePretraining(TrainingTask):
    """Contrastive pretraining task.

    This class supports contrastive pretraining with ``N`` modalities of data. It
    allows the sharing of encoders, heads, and postprocessors across modalities.
    It also supports computing the contrastive loss between specified pairs of
    modalities, as well as training auxiliary tasks alongside the main contrastive
    pretraining task.

    Parameters
    ----------
    encoders : dict[str, torch.nn.Module]
        A dictionary of encoders. The keys can be any string, including the names of
        any supported modalities. If the keys are not supported modalities, the
        ``modality_module_mapping`` parameter must be provided to map the encoders to
        specific modalities. The encoders are expected to take a dictionary of input
        values and return a list-like object with the first element being the encoded
        values. This first element is passed on to the heads or postprocessors and
        the remaining elements are ignored.
    heads : Optional[dict[str, Union[torch.nn.Module, dict[str, torch.nn.Module]]]], optional, default=None
        A dictionary of modules that process the encoder outputs, usually projection
        heads. If the keys do not correspond to the name of a supported modality,
        the ``modality_module_mapping`` parameter must be provided. If any of the values
        are dictionaries, they will be wrapped in a :py:class:`torch.nn.Sequential`
        module. All head modules are expected to take a single input tensor and
        return a single output tensor.
    postprocessors : Optional[dict[str, Union[torch.nn.Module, dict[str, torch.nn.Module]]]], optional, default=None
        A dictionary of modules that process the head outputs. If the keys do not
        correspond to the name of a supported modality, the `modality_module_mapping`
        parameter must be provided. If any of the values are dictionaries, they will
        be wrapped in a `nn.Sequential` module. All postprocessor modules are expected
        to take a single input tensor and return a single output tensor.
    modality_module_mapping : Optional[dict[str, ModuleKeySpec]], optional, default=None
        A dictionary mapping modalities to encoders, heads, and postprocessors.
        Useful for reusing the same instance of a module across multiple modalities.
    optimizer : Optional[partial[torch.optim.Optimizer]], optional, default=None
        The optimizer to use for training. This is expected to be a :py:func:`~functools.partial`
        function that takes the model parameters as the only required argument.
        If not provided, training will continue without an optimizer.
    lr_scheduler : Optional[Union[dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]], partial[torch.optim.lr_scheduler.LRScheduler]]], optional, default=None
        The learning rate scheduler to use for training. This can be a
        :py:func:`~functools.partial` function that takes the optimizer as the only
        required argument or a dictionary with a ``'scheduler'`` key that specifies
        the scheduler and an optional ``'extras'`` key that specifies additional
        arguments to pass to the scheduler. If not provided, the learning rate will
        not be adjusted during training.
    init_logit_scale : float, optional, default=1 / 0.07
        The initial value of the logit scale parameter. This is the log of the scale
        factor applied to the logits before computing the contrastive loss.
    max_logit_scale : float, optional, default=100
        The maximum value of the logit scale parameter. The logit scale parameter
        is clamped to the range ``[0, log(max_logit_scale)]``.
    learnable_logit_scale : bool, optional, default=True
        Whether the logit scale parameter is learnable. If set to False, the logit
        scale parameter is treated as a constant.
    loss : Optional[torch.nn.Module], optional, default=None
        The loss function to use.
    modality_loss_pairs : Optional[list[LossPairSpec]], optional, default=None
        A list of pairs of modalities to compute the contrastive loss between and
        the weight to apply to each pair.
    auxiliary_tasks : dict[str, AuxiliaryTaskSpec], optional, default=None
        Auxiliary tasks to run alongside the main contrastive pretraining task.

        - The auxiliary task module is expected to be a partially-initialized instance
          of a :py:class:`~lightning.pytorch.core.LightningModule` created using
          :py:func:`functools.partial`, such that an initialized encoder can be
          passed as the only argument.
        - The ``modality`` parameter specifies the modality of the encoder to use
          for the auxiliary task. The ``loss_weight`` parameter specifies the weight
          to apply to the auxiliary task loss.
    log_auxiliary_tasks_loss : bool, optional, default=False
        Whether to log the loss of auxiliary tasks to the main logger.
    compute_validation_loss : bool, optional, default=True
        Whether to compute the validation loss if a validation dataloader is provided.
        The loss function must be provided to compute the validation loss.
    compute_test_loss : bool, optional, default=True
        Whether to compute the test loss if a test dataloader is provided. The loss
        function must be provided to compute the test loss.
    evaluation_tasks : Optional[dict[str, EvaluationSpec]], optional, default=None
        Evaluation tasks to run during validation, while training, and during testing.

    Raises
    ------
    ValueError

        - If the loss function is not provided and either the validation or test loss
          needs to be computed.
        - If the given modality is not supported.
        - If the encoder, head, or postprocessor is not mapped to a modality.
        - If an unsupported modality is found in the loss pair specification.
        - If an unsupported modality is found in the auxiliary tasks.
        - If the auxiliary task is not a partial function.
        - If the evaluation task is not an instance of :py:class:`~mmlearn.tasks.hooks.EvaluationHooks`.

    """  # noqa: W505

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        encoders: dict[str, nn.Module],
        heads: Optional[dict[str, Union[nn.Module, dict[str, nn.Module]]]] = None,
        postprocessors: Optional[
            dict[str, Union[nn.Module, dict[str, nn.Module]]]
        ] = None,
        modality_module_mapping: Optional[dict[str, ModuleKeySpec]] = None,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        init_logit_scale: float = 1 / 0.07,
        max_logit_scale: float = 100,
        learnable_logit_scale: bool = True,
        loss: Optional[nn.Module] = None,
        modality_loss_pairs: Optional[list[LossPairSpec]] = None,
        auxiliary_tasks: Optional[dict[str, AuxiliaryTaskSpec]] = None,
        log_auxiliary_tasks_loss: bool = False,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[dict[str, EvaluationSpec]] = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss,
            compute_validation_loss=compute_validation_loss,
            compute_test_loss=compute_test_loss,
        )

        self.save_hyperparameters(
            ignore=[
                "encoders",
                "heads",
                "postprocessors",
                "modality_module_mapping",
                "loss",
                "auxiliary_tasks",
                "evaluation_tasks",
                "modality_loss_pairs",
            ]
        )

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
        modality_encoder_mapping: dict[str, Optional[str]] = {}
        modality_head_mapping: dict[str, Optional[str]] = {}
        modality_postprocessor_mapping: dict[str, Optional[str]] = {}
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

        self._available_modalities: list[Modality] = [
            Modalities.get_modality(modality_key)
            for modality_key in modality_encoder_mapping
        ]
        assert len(self._available_modalities) >= 2, (
            "Expected at least two modalities to be available. "
        )

        #: A :py:class:`~torch.nn.ModuleDict`, where the keys are the names of the
        #: modalities and the values are the encoder modules.
        self.encoders = nn.ModuleDict(
            {
                Modalities.get_modality(modality_key).name: encoders[encoder_key]
                for modality_key, encoder_key in modality_encoder_mapping.items()
                if encoder_key is not None
            }
        )

        #: A :py:class:`~torch.nn.ModuleDict`, where the keys are the names of the
        #: modalities and the values are the projection head modules. This can be
        #: ``None`` if no heads modules are provided.
        self.heads = None
        if heads is not None:
            self.heads = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key).name: heads[head_key]
                    if isinstance(heads[head_key], nn.Module)
                    else nn.Sequential(*heads[head_key].values())
                    for modality_key, head_key in modality_head_mapping.items()
                    if head_key is not None and head_key in heads
                }
            )

        #: A :py:class:`~torch.nn.ModuleDict`, where the keys are the names of the
        #: modalities and the values are the postprocessor modules. This can be
        #: ``None`` if no postprocessor modules are provided.
        self.postprocessors = None
        if postprocessors is not None:
            self.postprocessors = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key).name: postprocessors[
                        postprocessor_key
                    ]
                    if isinstance(postprocessors[postprocessor_key], nn.Module)
                    else nn.Sequential(*postprocessors[postprocessor_key].values())
                    for modality_key, postprocessor_key in modality_postprocessor_mapping.items()
                    if postprocessor_key is not None
                    and postprocessor_key in postprocessors
                }
            )

        # set up logit scaling
        log_logit_scale = torch.ones([]) * np.log(init_logit_scale)
        self.max_logit_scale = max_logit_scale
        self.learnable_logit_scale = learnable_logit_scale

        if self.learnable_logit_scale:
            self.log_logit_scale = torch.nn.Parameter(
                log_logit_scale, requires_grad=True
            )
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

        # set up contrastive loss pairs
        if modality_loss_pairs is None:
            modality_loss_pairs = [
                LossPairSpec(modalities=(m1.name, m2.name))
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

        #: A list :py:class:`LossPairSpec` instances specifying the pairs of
        #: modalities to compute the contrastive loss between and the weight to
        #: apply to each pair.
        self.modality_loss_pairs = modality_loss_pairs

        # set up auxiliary tasks
        self.aux_task_specs = auxiliary_tasks or {}

        #: A dictionary of auxiliary tasks to run alongside the main contrastive
        #: pretraining task.
        self.auxiliary_tasks: dict[str, L.LightningModule] = {}
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
                self.encoders[Modalities.get_modality(task_spec.modality).name]
            )

        self.log_auxiliary_tasks_loss = log_auxiliary_tasks_loss

        if evaluation_tasks is not None:
            for eval_task_spec in evaluation_tasks.values():
                if not isinstance(eval_task_spec.task, EvaluationHooks):
                    raise TypeError(
                        f"Expected {eval_task_spec.task} to be an instance of `EvaluationHooks` "
                        f"but got {type(eval_task_spec.task)}."
                    )

        #: A dictionary of evaluation tasks to run during validation, while training,
        #: or during testing.
        self.evaluation_tasks = evaluation_tasks

    def encode(
        self, inputs: dict[str, Any], modality: Modality, normalize: bool = False
    ) -> torch.Tensor:
        """Encode the input values for the given modality.

        Parameters
        ----------
        inputs : dict[str, Any]
            Input values.
        modality : Modality
            The modality to encode.
        normalize : bool, optional, default=False
            Whether to apply L2 normalization to the output (after the head and
            postprocessor layers, if present).

        Returns
        -------
        torch.Tensor
            The encoded values for the specified modality.
        """
        output = self.encoders[modality.name](inputs)[0]

        if self.heads and modality.name in self.heads:
            output = self.heads[modality.name](output)

        if normalize:
            output = torch.nn.functional.normalize(output, p=2, dim=-1)

        if self.postprocessors and modality.name in self.postprocessors:
            output = self.postprocessors[modality.name](output)

        return output

    def forward(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            The input tensors to encode.

        Returns
        -------
        dict[str, torch.Tensor]
            The encodings for each modality.
        """
        outputs = {
            modality.embedding: self.encode(inputs, modality, normalize=True)
            for modality in self._available_modalities
            if modality.name in inputs
        }

        if not all(
            output.size(-1) == list(outputs.values())[0].size(-1)
            for output in outputs.values()
        ):
            raise ValueError("Expected all model outputs to have the same dimension.")

        return outputs

    def _compute_loss(
        self, batch: dict[str, Any], batch_idx: int, outputs: dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return the sum of the contrastive loss and the auxiliary task losses."""
        if self.loss_fn is None:
            return None

        contrastive_loss = self.loss_fn(
            outputs,
            batch["example_ids"],
            self.log_logit_scale.exp(),
            self.modality_loss_pairs,
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
                        f"train/{task_name}_loss", auxiliary_task_loss, sync_dist=True
                    )

        if not auxiliary_losses:
            return contrastive_loss

        return torch.stack(auxiliary_losses).sum() + contrastive_loss

    def on_train_epoch_start(self) -> None:
        """Prepare for the training epoch.

        This method sets the modules to training mode.
        """
        self.encoders.train()
        if self.heads:
            self.heads.train()
        if self.postprocessors:
            self.postprocessors.train()

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
        outputs = self(batch)

        with torch.no_grad():
            self.log_logit_scale.clamp_(0, math.log(self.max_logit_scale))

        loss = self._compute_loss(batch, batch_idx, outputs)

        if loss is None:
            raise ValueError("The loss function must be provided for training.")

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "train/logit_scale",
            self.log_logit_scale.exp(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare for the validation epoch.

        This method sets the modules to evaluation mode and calls the
        ``on_evaluation_epoch_start`` method of each evaluation task.
        """
        self._on_eval_epoch_start("val")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        Optional[torch.Tensor]
            The loss for the batch or ``None`` if the loss function is not provided.
        """
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level metrics at the end of the validation epoch."""
        self._on_eval_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        """Prepare for the test epoch."""
        self._on_eval_epoch_start("test")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single test step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        Optional[torch.Tensor]
            The loss for the batch or ``None`` if the loss function is not provided.
        """
        return self._shared_eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self) -> None:
        """Compute and log epoch-level metrics at the end of the test epoch."""
        self._on_eval_epoch_end("test")

    def _on_eval_epoch_start(self, eval_type: Literal["val", "test"]) -> None:
        """Prepare for the evaluation epoch."""
        self.encoders.eval()
        if self.heads:
            self.heads.eval()
        if self.postprocessors:
            self.postprocessors.eval()
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.on_evaluation_epoch_start(self)

    def _shared_eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        eval_type: Literal["val", "test"],
    ) -> Optional[torch.Tensor]:
        """Run a single evaluation step.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The batch of data to process.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        Optional[torch.Tensor]
            The loss for the batch or ``None`` if the loss function is not provided.
        """
        loss: Optional[torch.Tensor] = None
        if (eval_type == "val" and self.compute_validation_loss) or (
            eval_type == "test" and self.compute_test_loss
        ):
            outputs = self(batch)
            loss = self._compute_loss(batch, batch_idx, outputs)
            if loss is not None and not self.trainer.sanity_checking:
                self.log(f"{eval_type}/loss", loss, prog_bar=True, sync_dist=True)

        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.evaluation_step(self, batch, batch_idx)

        return loss

    def _on_eval_epoch_end(self, eval_type: Literal["val", "test"]) -> None:
        """Compute and log epoch-level metrics at the end of the evaluation epoch."""
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.on_evaluation_epoch_end(self)
