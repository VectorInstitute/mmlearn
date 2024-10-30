"""IJEPA (Image Joint-Embedding Predictive Architecture) pretraining task."""

from typing import Any, Callable, Dict, Optional

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn

from mmlearn.datasets.core.modalities import Modalities
from mmlearn.datasets.processors.masking import IJEPAMaskGenerator, apply_masks
from mmlearn.datasets.processors.transforms import repeat_interleave_batch
from mmlearn.modules.ema import ExponentialMovingAverage
from mmlearn.modules.encoders.vision import VisionTransformer


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
    optimizer : Optional[torch.optim.Optimizer], optional
        Optimizer, by default None.
    lr_scheduler : Optional[Any], optional
        Learning rate scheduler, by default None.
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
    checkpoint_path : str, optional
        Path to a pre-trained model checkpoint, by default

    """

    def __init__(
        self,
        encoder: VisionTransformer,
        predictor: VisionTransformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        ema_anneal_end_step: int = 1000,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
    ):
        super().__init__()

        self.mask_generator = IJEPAMaskGenerator()

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.loss_fn = loss_fn if loss_fn is not None else F.smooth_l1_loss

        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

        self.current_step = 0
        self.total_steps = None

        self.encoder = encoder
        self.predictor = predictor

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
        images = batch[Modalities.RGB.name]

        # Generate masks
        mask_info = self.mask_generator()

        # Extract masks
        encoder_masks = mask_info["encoder_masks"]
        predictor_masks = mask_info["predictor_masks"]

        # Forward pass through target encoder to get h
        with torch.no_grad():
            h = self.target_encoder(images)
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer."""
        # Define parameters for weight decay and non-weight decay groups
        parameters = [
            {
                "params": (
                    p
                    for p in self.encoder.parameters()
                    if (p.ndim >= 2) and p.requires_grad
                ),
                "weight_decay": self.optimizer.keywords.get("weight_decay", 0.0),
                "name": "encoder_weight_decay_params",
            },
            {
                "params": (
                    p
                    for p in self.encoder.parameters()
                    if (p.ndim < 2) and p.requires_grad
                ),
                "weight_decay": 0.0,  # No weight decay for bias or 1D params
                "name": "encoder_no_weight_decay_params",
            },
            {
                "params": (
                    p
                    for p in self.predictor.parameters()
                    if (p.ndim >= 2) and p.requires_grad
                ),
                "weight_decay": self.optimizer.keywords.get("weight_decay", 0.0),
                "name": "predictor_weight_decay_params",
            },
            {
                "params": (
                    p
                    for p in self.predictor.parameters()
                    if (p.ndim < 2) and p.requires_grad
                ),
                "weight_decay": 0.0,  # No weight decay for bias or 1D params
                "name": "predictor_no_weight_decay_params",
            },
        ]

        # Initialize optimizer dynamically from the class attribute
        if self.optimizer is None:
            rank_zero_warn(
                "Optimizer not provided. Training will continue without an optimizer."
            )
            return None

        optimizer = self.optimizer(parameters)

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected optimizer to be an instance of `torch.optim.Optimizer`, but got {type(optimizer)}."
            )

        # Initialize the learning rate scheduler if available
        lr_scheduler = None
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                if "scheduler" not in self.lr_scheduler:
                    raise ValueError(
                        "Expected 'scheduler' key in the lr_scheduler dict."
                    )

                lr_scheduler = self.lr_scheduler["scheduler"](optimizer)
                lr_scheduler_dict = {"scheduler": lr_scheduler}
                if "extras" in self.lr_scheduler:
                    lr_scheduler_dict.update(self.lr_scheduler["extras"])
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

            lr_scheduler = self.lr_scheduler(optimizer)
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                raise TypeError(
                    f"Expected lr_scheduler to be an instance of `torch.optim.lr_scheduler._LRScheduler`, but got {type(lr_scheduler)}."
                )

        # Return optimizer and optionally scheduler
        return (
            {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
            if lr_scheduler
            else {"optimizer": optimizer}
        )

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
