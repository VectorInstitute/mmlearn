"""Data2Vec task."""

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Literal, Optional, Union

import lightning as L  # noqa: N812
import torch
from hydra_zen import store
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn

from mmlearn.datasets.processors.masking import apply_masks
from mmlearn.modules.ema import ExponentialMovingAverage
from mmlearn.modules.losses.data2vec import Data2VecLoss


@dataclass
class ModuleKeySpec:
    """Module key specification for mapping modules to modalities."""

    encoder_key: Optional[str] = None
    head_key: Optional[str] = None
    postprocessor_key: Optional[str] = None


@dataclass
class EvaluationSpec:
    """Specification for an evaluation task."""

    task: Any  # `EvaluationHooks` expected
    run_on_validation: bool = True
    run_on_test: bool = True


@store(group="task", provider="mmlearn")
class Data2VecTask(L.LightningModule):
    """Data2Vec task.

    This class implements the Data2Vec self-supervised learning approach for a single
    modality. It can be used as an auxiliary task in multi-modal learning setups.

    Parameters
    ----------
    encoder : nn.Module
        The encoder for the modality.
    optimizer : partial[torch.optim.Optimizer], optional
        The optimizer to use for training.
    lr_scheduler : Union[
        Dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
        partial[torch.optim.lr_scheduler.LRScheduler]
    ], optional
        The learning rate scheduler to use for training.
    loss : Data2VecLoss, optional
        The loss function to use.
    ema_decay : float
        The initial decay value for EMA.
    ema_end_decay : float
        The final decay value for EMA.
    ema_anneal_end_step : int
        The number of steps to anneal the decay from `ema_decay` to `ema_end_decay`.
    mask_generator : Any
        The mask generator to use for creating masked inputs.
    compute_validation_loss : bool
        Whether to compute the validation loss.
    compute_test_loss : bool
        Whether to compute the test loss.
    evaluation_tasks : Dict[str, EvaluationSpec], optional
        Evaluation tasks to run during validation and testing.
    """

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
        mask_generator: Any = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[Dict[str, EvaluationSpec]] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss or Data2VecLoss()

        self.ema = ExponentialMovingAverage(
            self,
            ema_decay,
            ema_end_decay,
            ema_anneal_end_step,
        )

        self.mask_generator = mask_generator
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss
        self.evaluation_tasks = evaluation_tasks

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input values.

        Parameters
        ----------
        inputs : torch.Tensor
            The input values to encode.

        Returns
        -------
        torch.Tensor
            The encoded values.
        """
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            The input values to forward pass.

        Returns
        -------
        torch.Tensor
            The forward pass output.
        """
        return self.encode(inputs)

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data to compute the loss for.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        masked_input = apply_masks(batch, self.mask_generator())
        student_output = self.encode(masked_input)
        with torch.no_grad():
            teacher_output = self.ema.model.encode(batch)
        return self.loss_fn(student_output, teacher_output)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute the loss for the batch.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data to compute the loss for.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        loss = self._compute_loss(batch)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.ema.step(self)

        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare for the validation epoch."""
        self._on_eval_epoch_start("val")

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step.

        Parameters
        ----------
        batch : torch.Tensor
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

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        """Run a single test step.

        Parameters
        ----------
        batch : torch.Tensor
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
        batch: torch.Tensor,
        batch_idx: int,
        eval_type: Literal["val", "test"],
    ) -> Optional[torch.Tensor]:
        """Run a single evaluation step.

        Parameters
        ----------
        batch : torch.Tensor
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
