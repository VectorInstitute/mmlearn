"""Data2Vec task."""

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F
from hydra_zen import store
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn

from mmlearn.datasets.core.modalities import Modalities, Modality
from mmlearn.datasets.processors.masking import (
    BlockwiseImagePatchMaskGenerator,
    apply_masks,
)
from mmlearn.modules.ema import ExponentialMovingAverage
from mmlearn.modules.losses.data2vec import Data2VecLoss


@dataclass
class EvaluationSpec:
    """Specification for an evaluation task."""

    task: Any  # `EvaluationHooks` expected
    run_on_validation: bool = True
    run_on_test: bool = True


@store(group="task", provider="mmlearn")
class Data2VecTask(L.LightningModule):
    """Data2Vec task implementation."""

    def __init__(
        self,
        encoder: nn.Module,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                Dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        loss: Optional[Data2VecLoss] = None,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 300000,
        mask_generator: Optional[BlockwiseImagePatchMaskGenerator] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[Dict[str, EvaluationSpec]] = None,
        average_top_k_layers: int = 6,
        target_instance_norm: bool = False,
        target_batch_norm: bool = False,
        target_layer_norm_last: bool = False,
        post_target_instance_norm: bool = False,
        post_target_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss or Data2VecLoss()

        self.ema = ExponentialMovingAverage(
            self.encoder,
            ema_decay,
            ema_end_decay,
            ema_anneal_end_step,
        )

        self.mask_generator = mask_generator
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss
        self.evaluation_tasks = evaluation_tasks

        # Data2Vec specific parameters
        self.average_top_k_layers = average_top_k_layers
        self.target_instance_norm = target_instance_norm
        self.target_batch_norm = target_batch_norm
        self.target_layer_norm_last = target_layer_norm_last
        self.post_target_instance_norm = post_target_instance_norm
        self.post_target_layer_norm = post_target_layer_norm

    def _get_teacher_targets(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Get teacher targets by averaging top k layers with normalization.

        Parameters
        ----------
        hidden_states : List[torch.Tensor]
            The hidden states to average.

        Returns
        -------
        torch.Tensor
            The averaged hidden states.
        """
        top_k_hidden_states = hidden_states[-self.average_top_k_layers :]

        if self.target_instance_norm or self.target_batch_norm:
            top_k_hidden_states = [
                val.permute(0, 2, 1) for val in top_k_hidden_states
            ]  # btc => bct

        if self.target_batch_norm:
            top_k_hidden_states = [
                F.batch_norm(
                    val.float(), running_mean=None, running_var=None, training=True
                )
                for val in top_k_hidden_states
            ]

        if self.target_instance_norm:
            top_k_hidden_states = [
                F.instance_norm(val.float()) for val in top_k_hidden_states
            ]

        if self.target_instance_norm or self.target_batch_norm:
            top_k_hidden_states = [
                val.permute(0, 2, 1) for val in top_k_hidden_states
            ]  # bct => btc

        if self.target_layer_norm_last:
            top_k_hidden_states = [
                F.layer_norm(val.float(), val.shape[-1:]) for val in top_k_hidden_states
            ]

        targets = sum(top_k_hidden_states) / len(top_k_hidden_states)

        if self.post_target_instance_norm:
            targets = targets.permute(0, 2, 1)
            targets = F.instance_norm(targets.float())
            targets = targets.permute(0, 2, 1)

        if self.post_target_layer_norm:
            targets = F.layer_norm(targets.float(), targets.shape[-1:])

        return targets

    def _compute_loss(self, batch: Dict[Union[str, Modality], Any]) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to compute the loss for.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        # Generate mask
        mask = self.mask_generator()

        # Apply mask to input (only supported for RGB modality)
        masked_input = {
            k: apply_masks(v, mask) if k == Modalities.RGB else v
            for k, v in batch.items()
        }

        # Get student output with masked input
        student_output = self(masked_input)
        student_hidden = student_output.last_hidden_state

        # Get teacher output with original input
        with torch.no_grad():
            teacher_output = self.ema.model(batch, output_hidden_states=True)
            teacher_targets = self._get_teacher_targets(teacher_output.hidden_states)

        # Get masked indices
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()

        # Compute loss only on masked positions
        return self.loss_fn(student_hidden[mask], teacher_targets[mask])

    def training_step(
        self, batch: Dict[Union[str, Modality], Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : Dict[Union[str, Modality], Any]
            The batch of data to compute the loss for.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.ema.step(self.encoder)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare for the validation epoch."""
        self._on_eval_epoch_start("val")

    def validation_step(
        self, batch: Dict[Union[str, Modality], Any], batch_idx: int
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
        self, batch: Dict[Union[str, Modality], Any], batch_idx: int
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

    def _shared_eval_step(
        self,
        batch: Dict[Union[str, Modality], Any],
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
        eval_type : Literal["val", "test"]
            The type of evaluation to run.

        Returns
        -------
        torch.Tensor or None
            The loss for the batch or None if the loss function is not provided.
        """
        loss = None

        if (eval_type == "val" and self.compute_validation_loss) or (
            eval_type == "test" and self.compute_test_loss
        ):
            loss = self._compute_loss(batch)
            self.log(f"{eval_type}/loss", loss, prog_bar=True, sync_dist=True)

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

    def _on_eval_epoch_start(self, eval_type: Literal["val", "test"]) -> None:
        """Prepare for the evaluation epoch.

        Parameters
        ----------
        eval_type : Literal["val", "test"]
            The type of evaluation to run.
        """
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.on_evaluation_epoch_start(self)

    def _on_eval_epoch_end(self, eval_type: Literal["val", "test"]) -> None:
        """Compute and log epoch-level metrics at the end of the evaluation epoch.

        Parameters
        ----------
        eval_type : Literal["val", "test"]
            The type of evaluation to run.
        """
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    results = task_spec.task.on_evaluation_epoch_end(self)
                    if results:
                        for key, value in results.items():
                            self.log(f"{eval_type}/{key}", value)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler.

        Returns
        -------
        OptimizerLRScheduler
            The optimizer and learning rate scheduler.
        """
        if self.optimizer is None:
            rank_zero_warn(
                "Optimizer not provided. Training will continue without an optimizer. "
                "LR scheduler will not be used."
            )
            return None

        weight_decay: Optional[float] = self.optimizer.keywords.get(
            "weight_decay", None
        )
        if weight_decay is None:
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

                if param.ndim < 2:
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

        if self.lr_scheduler is None:
            return optimizer

        if isinstance(self.lr_scheduler, dict):
            scheduler = self.lr_scheduler["scheduler"](optimizer)
            lr_scheduler_dict = {"scheduler": scheduler}
            lr_scheduler_dict.update(self.lr_scheduler.get("extras", {}))
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

        scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [scheduler]
