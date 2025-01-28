"""Implementation of Data2vec loss function."""

import math
from typing import Optional

import torch
from hydra_zen import store
from torch import nn
from torch.nn.functional import mse_loss, smooth_l1_loss


@store(group="modules/losses", provider="mmlearn")
class Data2VecLoss(nn.Module):
    """Data2Vec loss function.

    Parameters
    ----------
    beta : float, optional, default=0
        Specifies the beta parameter for smooth L1 loss. If ``0``, MSE loss is used.
    loss_scale : Optional[float], optional, default=None
        Scaling factor for the loss. If None, uses ``1 / sqrt(embedding_dim)``.
    reduction : str, optional, default='none'
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.

    Raises
    ------
    ValueError
        If the reduction mode is not supported.
    """

    def __init__(
        self,
        beta: float = 0,
        loss_scale: Optional[float] = None,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.loss_scale = loss_scale
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Data2Vec loss.

        Parameters
        ----------
        x : torch.Tensor
            Predicted embeddings of shape ``(batch_size, num_patches, embedding_dim)``.
        y : torch.Tensor
            Target embeddings of shape ``(batch_size, num_patches, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Data2Vec loss value.

        Raises
        ------
        ValueError
            If the shapes of x and y do not match.
        """
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x: {x.shape}, y: {y.shape}")

        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, y.size(-1))

        if self.beta == 0:
            loss = mse_loss(x, y, reduction="none")
        else:
            loss = smooth_l1_loss(x, y, reduction="none", beta=self.beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        loss = loss * scale

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        # 'none'
        return loss.view(x.size(0), -1).sum(1)
