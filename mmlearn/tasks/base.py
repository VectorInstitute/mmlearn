"""Base class for all tasks in mmlearn that require training."""

import inspect
from functools import partial
from typing import Any, Optional, Union

import lightning as L  # noqa: N812
import torch
import torch.distributed
import torch.distributed.nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_warn


class TrainingTask(L.LightningModule):
    """Base class for all tasks in mmlearn that require training.

    Parameters
    ----------
    optimizer : Optional[partial[torch.optim.Optimizer]], optional, default=None
        The optimizer to use for training. This is expected to be a partial function,
        created using `functools.partial`, that takes the model parameters as the
        only required argument. If not provided, training will continue without an
        optimizer.
    lr_scheduler : Optional[Union[dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]], partial[torch.optim.lr_scheduler.LRScheduler]]], optional, default=None
        The learning rate scheduler to use for training. This can be a partial function
        that takes the optimizer as the only required argument or a dictionary with
        a `scheduler` key that specifies the scheduler and an optional `extras` key
        that specifies additional arguments to pass to the scheduler. If not provided,
        the learning rate will not be adjusted during training.
    loss_fn : Optional[torch.nn.Module], optional, default=None
        Loss function to use for training.
    compute_validation_loss : bool, optional, default=True
        Whether to compute the validation loss if a validation dataloader is provided.
        The loss function must be provided to compute the validation loss.
    compute_test_loss : bool, optional, default=True
        Whether to compute the test loss if a test dataloader is provided. The loss
        function must be provided to compute the test loss.

    Raises
    ------
    ValueError
        If the loss function is not provided and either the validation or test loss
        needs to be computed.
    """  # noqa: W505

    def __init__(
        self,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
    ):
        super().__init__()
        if loss_fn is None and (compute_validation_loss or compute_test_loss):
            raise ValueError(
                "Loss function must be provided to compute validation or test loss."
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

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
                lr_scheduler_dict: dict[
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
