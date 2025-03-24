"""IJEPA (Image Joint-Embedding Predictive Architecture) pretraining task."""

from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from hydra_zen import store

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.masking import IJEPAMaskGenerator, apply_masks
from mmlearn.datasets.processors.transforms import repeat_interleave_batch
from mmlearn.modules.ema import ExponentialMovingAverage
from mmlearn.modules.encoders.vision import (
    VisionTransformer,
    VisionTransformerPredictor,
)
from mmlearn.tasks.base import TrainingTask


# NOTE: setting zen_partial is necessary for adding the `_partial_` attribute to the
# configuration, so that this task can be partially instantiated, if needed. This is
# necessary when combining the task with other tasks in a multitask setting.
@store(group="task", provider="mmlearn", zen_partial=False)
class IJEPA(TrainingTask):
    """Pretraining module for IJEPA.

    This class implements the IJEPA (Image Joint-Embedding Predictive Architecture)
    pretraining task using PyTorch Lightning. It trains an encoder and a predictor to
    reconstruct masked regions of an image based on its unmasked context.

    Parameters
    ----------
    encoder : VisionTransformer
        Vision transformer encoder.
    predictor : VisionTransformerPredictor
        Vision transformer predictor.
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
    ema_decay : float, optional, default=0.996
        Initial momentum for EMA of target encoder.
    ema_decay_end : float, optional, default=1.0
        Final momentum for EMA of target encoder.
    ema_anneal_end_step : int, optional, default=1000
        Number of steps to anneal EMA momentum to ``ema_decay_end``.
    loss_fn : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional
        Loss function to use. If not provided, defaults to
        :py:func:`~torch.nn.functional.smooth_l1_loss`.
    compute_validation_loss : bool, optional, default=True
        Whether to compute validation loss.
    compute_test_loss : bool, optional, default=True
        Whether to compute test loss.
    """  # noqa: W505

    def __init__(
        self,
        encoder: VisionTransformer,
        predictor: VisionTransformerPredictor,
        modality: str = "RGB",
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        ema_anneal_end_step: int = 1000,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn if loss_fn is not None else F.smooth_l1_loss,
            compute_validation_loss=compute_validation_loss,
            compute_test_loss=compute_test_loss,
        )
        self.modality = Modalities.get_modality(modality)
        self.mask_generator = IJEPAMaskGenerator()

        self.encoder = encoder
        self.predictor = predictor

        self.predictor.num_patches = encoder.patch_embed.num_patches
        self.predictor.embed_dim = encoder.embed_dim
        self.predictor.num_heads = encoder.num_heads

        self.target_encoder = ExponentialMovingAverage(
            self.encoder, ema_decay, ema_decay_end, ema_anneal_end_step
        )

    def configure_model(self) -> None:
        """Configure the model."""
        self.target_encoder.configure_model(self.device)

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform exponential moving average update of target encoder.

        This is done right after the ``optimizer.step()`, which comes just before
        ``optimizer.zero_grad()`` to account for gradient accumulation.
        """
        if self.target_encoder is not None:
            self.target_encoder.step(self.encoder)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return self._shared_step(batch, batch_idx, step_type="train")

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Optional[torch.Tensor]
            Loss value or ``None`` if no loss is computed.
        """
        return self._shared_step(batch, batch_idx, step_type="val")

    def test_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single test step.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        Optional[torch.Tensor]
            Loss value or ``None`` if no loss is computed
        """
        return self._shared_step(batch, batch_idx, step_type="test")

    def on_validation_epoch_start(self) -> None:
        """Prepare for the validation epoch."""
        self._on_eval_epoch_start("val")

    def on_validation_epoch_end(self) -> None:
        """Actions at the end of the validation epoch."""
        self._on_eval_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        """Prepare for the test epoch."""
        self._on_eval_epoch_start("test")

    def on_test_epoch_end(self) -> None:
        """Actions at the end of the test epoch."""
        self._on_eval_epoch_end("test")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add relevant EMA state to the checkpoint.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            The state dictionary to save the EMA state to.
        """
        if self.target_encoder is not None:
            checkpoint["ema_params"] = {
                "decay": self.target_encoder.decay,
                "num_updates": self.target_encoder.num_updates,
            }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore EMA state from the checkpoint.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            The state dictionary to restore the EMA state from.
        """
        if "ema_params" in checkpoint and self.target_encoder is not None:
            ema_params = checkpoint.pop("ema_params")
            self.target_encoder.decay = ema_params["decay"]
            self.target_encoder.num_updates = ema_params["num_updates"]

            self.target_encoder.restore(self.encoder)

    def _shared_step(
        self, batch: dict[str, Any], batch_idx: int, step_type: str
    ) -> Optional[torch.Tensor]:
        images = batch[self.modality.name]

        # Generate masks
        batch_size = images.size(0)
        mask_info = self.mask_generator(batch_size=batch_size)

        # Extract masks and move to device
        device = images.device
        encoder_masks = [mask.to(device) for mask in mask_info["encoder_masks"]]
        predictor_masks = [mask.to(device) for mask in mask_info["predictor_masks"]]

        # Forward pass through target encoder to get h
        with torch.no_grad():
            h = self.target_encoder.model(batch)[0]
            h = F.layer_norm(h, h.size()[-1:])
            h_masked = apply_masks(h, predictor_masks)
            h_masked = repeat_interleave_batch(
                h_masked, images.size(0), repeat=len(encoder_masks)
            )

        # Forward pass through encoder with encoder_masks
        batch[self.modality.mask] = encoder_masks
        z = self.encoder(batch)[0]

        # Pass z through predictor with encoder_masks and predictor_masks
        z_pred = self.predictor(z, encoder_masks, predictor_masks)

        if step_type == "train":
            self.log("train/ema_decay", self.target_encoder.decay, prog_bar=True)

        if self.loss_fn is not None and (
            step_type == "train"
            or (step_type == "val" and self.compute_validation_loss)
            or (step_type == "test" and self.compute_test_loss)
        ):
            # Compute loss between z_pred and h_masked
            loss = self.loss_fn(z_pred, h_masked)

            # Log loss
            self.log(f"{step_type}/loss", loss, prog_bar=True, sync_dist=True)

            return loss

        return None

    def _on_eval_epoch_start(self, step_type: str) -> None:
        """Initialize states or configurations at the start of an evaluation epoch.

        Parameters
        ----------
        step_type : str
            Type of the evaluation phase ("val" or "test").
        """
        if (
            step_type == "val"
            and self.compute_validation_loss
            or step_type == "test"
            and self.compute_test_loss
        ):
            self.log(f"{step_type}/start", 1, prog_bar=True, sync_dist=True)

    def _on_eval_epoch_end(self, step_type: str) -> None:
        """Finalize states or logging at the end of an evaluation epoch.

        Parameters
        ----------
        step_type : str
            Type of the evaluation phase ("val" or "test").
        """
        if (
            step_type == "val"
            and self.compute_validation_loss
            or step_type == "test"
            and self.compute_test_loss
        ):
            self.log(f"{step_type}/end", 1, prog_bar=True, sync_dist=True)
