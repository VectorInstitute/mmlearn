"""Linear warmup cosine annealing learning rate scheduler."""

from hydra_zen import MISSING, store
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


@store(  # type: ignore[misc]
    group="modules/lr_schedulers",
    provider="mmlearn",
    zen_partial=True,
    warmup_steps=MISSING,
    max_steps=MISSING,
)
def linear_warmup_cosine_annealing_lr(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    start_factor: float = 1 / 3,
    eta_min: float = 0.0,
    last_epoch: int = -1,
) -> LRScheduler:
    """Create a linear warmup cosine annealing learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Maximum number of iterations for linear warmup.
    max_steps : int
        Maximum number of iterations.
    start_factor : float, default=1/3
        Multiplicative factor for the learning rate at the start of the warmup phase.
    eta_min : float, default=0
        Minimum learning rate.
    last_epoch : int, default=-1
        The index of last epoch. If set to -1, it initializes the learning rate
        as the base learning rate

    Returns
    -------
    LRScheduler
        The learning rate scheduler.

    Raises
    ------
    ValueError
        If `warmup_steps` is greater than or equal to `max_steps` or if `warmup_steps`
        is less than or equal to 0.
    """
    if warmup_steps >= max_steps:
        raise ValueError(
            "Expected `warmup_steps` to be less than `max_steps` but got "
            f"`warmup_steps={warmup_steps}` and `max_steps={max_steps}`."
        )
    if warmup_steps <= 0:
        raise ValueError(
            "Expected `warmup_steps` to be positive but got "
            f"`warmup_steps={warmup_steps}`."
        )

    linear_lr = LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_steps,
        last_epoch=last_epoch,
    )
    cosine_lr = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=eta_min,
        last_epoch=last_epoch,
    )
    return SequentialLR(
        optimizer,
        schedulers=[linear_lr, cosine_lr],
        milestones=[warmup_steps],
        last_epoch=last_epoch,
    )
