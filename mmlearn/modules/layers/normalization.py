"""Normalization layers."""

import torch
from hydra_zen import store


@store(group="modules/layers", provider="mmlearn")
class L2Norm(torch.nn.Module):
    """L2 normalization module.

    Parameters
    ----------
    dim : int
        The dimension along which to normalize.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_sz, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor of shape (batch_sz, seq_len, dim).
        """
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)
