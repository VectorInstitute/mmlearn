"""IJEPA (Image Joint-Embedding Predictive Architecture) pretraining task."""

from typing import Any, Callable, Dict, Optional

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from hydra.utils import instantiate  # Import instantiate
from hydra_zen import store

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.masking import IJEPAMaskGenerator, apply_masks
from mmlearn.datasets.processors.transforms import repeat_interleave_batch
from mmlearn.modules.ema import ExponentialMovingAverage
from mmlearn.modules.encoders.vision import VisionTransformer


@store(group="task", provider="mmlearn")
class IJEPA(L.LightningModule):
    """Pretraining module for IJEPA.

    This class implements the IJEPA (Image Joint-Embedding Predictive Architecture)
    pretraining task using PyTorch Lightning. It trains an encoder and a predictor to
    reconstruct masked regions of an image based on its unmasked context.

    Parameters
    ----------
    encoder : VisionTransformer
        Vision transformer encoder.
    predictor : VisionTransformer
        Vision transformer predictor.
    optimizer_cfg : Optional[Dict[str, Any]], optional
        Optimizer configuration dictionary, by default None.
    lr_scheduler_cfg : Optional[Dict[str, Any]], optional
        Learning rate scheduler configuration dictionary, by default None.
    ema_decay : float, optional
        Initial momentum for EMA of target encoder, by default 0.996.
    ema_decay_end : float, optional
        Final momentum for EMA of target encoder, by default 1.0.
    ema_anneal_end_step : int, optional
        Number of steps to anneal EMA momentum to `ema_decay_end`, by default 1000.
    loss_fn : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional
        Loss function to use, by default None.
    compute_validation_loss : bool, optional
        Whether to compute validation loss, by default True.
    compute_test_loss : bool, optional
        Whether to compute test loss, by default True.
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        predictor: VisionTransformer,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        lr_scheduler_cfg: Optional[Dict[str, Any]] = None,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        ema_anneal_end_step: int = 1000,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
    ):
        super().__init__()

        self.mask_generator = IJEPAMaskGenerator()

        self.optimizer_cfg = optimizer_cfg  # Store optimizer config
        self.lr_scheduler_cfg = lr_scheduler_cfg  # Store lr_scheduler config
        self.loss_fn = loss_fn if loss_fn is not None else F.smooth_l1_loss

        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

        self.current_step = 0
        self.total_steps = None

        self.encoder = encoder
        self.predictor = predictor

        self.predictor.num_patches = encoder.patch_embed.num_patches
        self.predictor.embed_dim = encoder.embed_dim
        self.predictor.num_heads = encoder.num_heads

        self.ema = ExponentialMovingAverage(
            encoder,
            ema_decay,
            ema_decay_end,
            ema_anneal_end_step,
            device_id=self.device,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._shared_step(batch, batch_idx, step_type="train")

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step."""
        return self._shared_step(batch, batch_idx, step_type="val")

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single test step."""
        return self._shared_step(batch, batch_idx, step_type="test")

    def _shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        step_type: str,
    ) -> Optional[torch.Tensor]:
        images = batch[Modalities.RGB]

        # Generate masks
        batch_size = images.size(0)
        mask_info = self.mask_generator(batch_size=batch_size)

        # Extract masks and move to device
        device = images.device
        encoder_masks = [mask.to(device) for mask in mask_info["encoder_masks"]]
        predictor_masks = [mask.to(device) for mask in mask_info["predictor_masks"]]

        # Forward pass through target encoder to get h
        with torch.no_grad():
            h = self.ema.model(images)
            h = F.layer_norm(h, h.size()[-1:])
            h_masked = apply_masks(h, predictor_masks)
            h_masked = repeat_interleave_batch(
                h_masked, images.size(0), repeat=len(encoder_masks)
            )

        # Forward pass through encoder with encoder_masks
        z = self.encoder(images, masks=encoder_masks)

        # Pass z through predictor with encoder_masks and predictor_masks
        z_pred = self.predictor(z, encoder_masks, predictor_masks)

        # Compute loss between z_pred and h_masked
        loss = self.loss_fn(z_pred, h_masked)

        # Log loss
        self.log(
            f"{step_type}/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        if step_type == "train":
            # EMA update of target encoder
            self.ema.step(self.encoder)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler."""
        weight_decay_value = 0.05  # Desired weight decay

        # Define parameters for weight decay and non-weight decay groups
        parameters = [
            {
                "params": [
                    p
                    for p in self.encoder.parameters()
                    if p.ndim >= 2 and p.requires_grad
                ],
                "weight_decay": weight_decay_value,
            },
            {
                "params": [
                    p
                    for p in self.encoder.parameters()
                    if p.ndim < 2 and p.requires_grad
                ],
                "weight_decay": 0.0,  # No weight decay for bias or 1D params
            },
            {
                "params": [
                    p
                    for p in self.predictor.parameters()
                    if p.ndim >= 2 and p.requires_grad
                ],
                "weight_decay": weight_decay_value,
            },
            {
                "params": [
                    p
                    for p in self.predictor.parameters()
                    if p.ndim < 2 and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(parameters, lr=0.001)

        # Instantiate the learning rate scheduler if provided
        lr_scheduler = None
        if self.lr_scheduler_cfg is not None:
            lr_scheduler = instantiate(self.lr_scheduler_cfg, optimizer=optimizer)

        # Return optimizer and scheduler in Lightning-compatible format
        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",  # Change to 'step' if desired
                    "frequency": 1,
                    # Add 'monitor' key if necessary, e.g., 'monitor': 'val_loss'
                },
            }
        return {"optimizer": optimizer}

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
