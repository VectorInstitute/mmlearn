"""IJEPA (Image Joint-Embedding Predictive Architecture) pretraining task."""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn

from mmlearn.datasets.core.modalities import Modalities
from mmlearn.datasets.processors.masking import IJEPAMaskGenerator, apply_masks
from mmlearn.datasets.processors.transforms import repeat_interleave_batch
from mmlearn.modules.encoders.vision import VisionTransformer


class IJEPAPretraining(L.LightningModule):
    """Pretraining module for IJEPA.

    This class implements the IJEPA (Image Joint-Embedding Predictive Architecture)
    pretraining task using PyTorch Lightning. It trains an encoder and a predictor to
    reconstruct masked regions of an image based on its unmasked context.

    Parameters
    ----------
    model_name : str
        Name of the Vision Transformer model to use.
    crop_size : int
        Size of the input image crop.
    patch_size : int
        Size of the image patches.
    pred_emb_dim : int
        Dimension of the predictor embeddings.
    pred_depth : int
        Depth of the predictor.
    optimizer : Optional[Any], optional
        Optimizer configuration, by default None.
    lr_scheduler : Optional[Any], optional
        Learning rate scheduler configuration, by default None.
    ema_momentum : float, optional
        Momentum for exponential moving average (EMA) of target encoder
        , by default 0.996.
    ema_momentum_end : float, optional
        Final momentum for EMA of target encoder, by default 1.0.
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
        model_name: str,
        crop_size: int,
        patch_size: int,
        pred_emb_dim: int,
        pred_depth: int,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        ema_momentum: float = 0.996,
        ema_momentum_end: float = 1.0,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        checkpoint_path: str = "",
    ):
        super().__init__()

        self.mask_generator = IJEPAMaskGenerator()

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.loss_fn = loss_fn if loss_fn is not None else F.smooth_l1_loss

        self.ema_momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end

        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

        self.current_step = 0
        self.total_steps = None

        self.encoder = VisionTransformer.__dict__[model_name](
            img_size=[crop_size], patch_size=patch_size
        )

        self.predictor = VisionTransformer.__dict__["vit_predictor"](
            num_patches=self.encoder.patch_embed.num_patches,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.encoder.num_heads,
        )

        self.target_encoder = copy.deepcopy(self.encoder)

        if checkpoint_path != "":
            self.encoder, self.predictor, self.target_encoder, _, _, _ = (
                self.load_checkpoint(
                    device=self.device,
                    checkpoint_path=checkpoint_path,
                    encoder=self.encoder,
                    predictor=self.predictor,
                    target_encoder=self.target_encoder,
                    opt=None,
                    scaler=None,
                )
            )

        # Freeze parameters of target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.target_encoder.to(self.device)

    def load_checkpoint(
        self,
        device: str,
        checkpoint_path: str,
        encoder: nn.Module,
        predictor: nn.Module,
        target_encoder: nn.Module,
        opt: Any,
        scaler: Any,
    ) -> Tuple[nn.Module, nn.Module, nn.Module, Any, Any, int]:
        """Load a pre-trained model from a checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
            epoch = checkpoint["epoch"]

            # loading encoder
            pretrained_dict = checkpoint["encoder"]
            msg = encoder.load_state_dict(pretrained_dict)
            print(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

            # loading predictor
            pretrained_dict = checkpoint["predictor"]
            msg = predictor.load_state_dict(pretrained_dict)
            print(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

            # loading target_encoder
            if target_encoder is not None:
                print(list(checkpoint.keys()))
                pretrained_dict = checkpoint["target_encoder"]
                msg = target_encoder.load_state_dict(pretrained_dict)
                print(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

            # loading optimizer
            opt.load_state_dict(checkpoint["opt"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            print(f"loaded optimizers from epoch {epoch}")
            del checkpoint

        except Exception as e:
            print(f"Encountered exception when loading checkpoint {e}")
            epoch = 0

        return encoder, predictor, target_encoder, opt, scaler, epoch

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encode(x, masks)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        images = batch[Modalities.RGB]

        # Generate masks
        mask_info = self.mask_generator()

        # Extract masks
        encoder_masks = mask_info["encoder_masks"]
        predictor_masks = mask_info["predictor_masks"]

        # Move images and masks to device
        images = images.to(self.device)
        encoder_masks = encoder_masks.to(self.device)
        predictor_masks = predictor_masks.to(self.device)

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
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # EMA update of target encoder
        self._update_target_encoder()

        return loss

    def _update_target_encoder(self) -> None:
        """Update the target encoder using exponential moving average (EMA)."""
        if self.total_steps is None:
            self.total_steps = self.trainer.estimated_stepping_batches
        current_step = self.trainer.global_step
        m = self.ema_momentum + (self.ema_momentum_end - self.ema_momentum) * (
            current_step / self.total_steps
        )
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer."""
        # Define parameters for weight decay and non-weight decay groups
        param_groups = [
            {
                "params": (p for n, p in self.encoder.named_parameters()
                           if ("bias" not in n) and (len(p.shape) != 1)),
                "weight_decay": self.optimizer_cfg.get("weight_decay", 0.0)
            },
            {
                "params": (p for n, p in self.predictor.named_parameters()
                           if ("bias" not in n) and (len(p.shape) != 1)),
                "weight_decay": self.optimizer_cfg.get("weight_decay", 0.0)
            },
            {
                "params": (p for n, p in self.encoder.named_parameters()
                           if ("bias" in n) or (len(p.shape) == 1)),
                "weight_decay": 0.0  # No weight decay for bias or 1D params
            },
            {
                "params": (p for n, p in self.predictor.named_parameters()
                           if ("bias" in n) or (len(p.shape) == 1)),
                "weight_decay": 0.0  # No weight decay for bias or 1D params
            }
        ]

        # Initialize optimizer dynamically from the class attribute
        if self.optimizer is None:
            rank_zero_warn("Optimizer not provided. Training will continue without an optimizer.")
            return None

        optimizer = self.optimizer(param_groups, **self.optimizer_cfg.get("params", {}))

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Expected optimizer to be an instance of `torch.optim.Optimizer`, but got {type(optimizer)}.")

        # Initialize the learning rate scheduler if available
        lr_scheduler = None
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                if "scheduler" not in self.lr_scheduler:
                    raise ValueError("Expected 'scheduler' key in the lr_scheduler dict.")

                lr_scheduler = self.lr_scheduler["scheduler"](optimizer)
                lr_scheduler_dict = {"scheduler": lr_scheduler}
                if "extras" in self.lr_scheduler:
                    lr_scheduler_dict.update(self.lr_scheduler["extras"])
                return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

            lr_scheduler = self.lr_scheduler(optimizer)
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                raise TypeError(f"Expected lr_scheduler to be an instance of `torch.optim.lr_scheduler._LRScheduler`, but got {type(lr_scheduler)}.")

        # Return optimizer and optionally scheduler
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} if lr_scheduler else {"optimizer": optimizer}


    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single validation step."""
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        """Run a single test step."""
        return self._shared_eval_step(batch, batch_idx, "test")

    def _shared_eval_step(
        self, batch: Dict[str, Any], batch_idx: int, eval_type: str
    ) -> Optional[torch.Tensor]:
        images = batch[Modalities.RGB]

        # Generate masks
        mask_info = self.mask_generator()

        # Extract masks
        encoder_masks = mask_info["encoder_masks"]
        predictor_masks = mask_info["predictor_masks"]

        # Move images and masks to device
        images = images.to(self.device)
        encoder_masks = encoder_masks.to(self.device)
        predictor_masks = predictor_masks.to(self.device)

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
            f"{eval_type}/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

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

    def _on_eval_epoch_start(self, eval_type: str) -> None:
        """Prepare for evaluation."""
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    task_spec.task.on_evaluation_epoch_start(self)

    def _on_eval_epoch_end(self, eval_type: str) -> None:
        """Handle the end of an evaluation epoch."""
        if self.evaluation_tasks:
            for task_spec in self.evaluation_tasks.values():
                if (eval_type == "val" and task_spec.run_on_validation) or (
                    eval_type == "test" and task_spec.run_on_test
                ):
                    results = task_spec.task.on_evaluation_epoch_end(self)
                    if results:
                        for key, value in results.items():
                            self.log(f"{eval_type}/{key}", value)
