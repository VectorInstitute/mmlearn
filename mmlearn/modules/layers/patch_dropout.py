"""Patch dropout layer."""

from typing import Optional

import torch


# modified from: https://github.com/yueliukth/PatchDropout/blob/main/scripts/patchdropout.py
class PatchDropout(torch.nn.Module):
    """Patch dropout layer.

    Drops patch tokens (after embedding and adding CLS token) from the input tensor.
    Usually used in vision transformers to reduce the number of tokens. [1]_

    Parameters
    ----------
    keep_rate : float, optional, default=0.5
        The proportion of tokens to keep.
    bias : Optional[float], optional, default=None
        The bias to add to the random noise before sorting.
    token_shuffling : bool, optional, default=False
        If True, the tokens are shuffled.

    References
    ----------
    .. [1] Liu, Y., Matsoukas, C., Strand, F., Azizpour, H., & Smith, K. (2023).
       Patchdropout: Economizing vision transformers using patch dropout. In Proceedings
       of the IEEE/CVF Winter Conference on Applications of Computer Vision
       (pp. 3953-3962).
    """

    def __init__(
        self,
        keep_rate: float = 0.5,
        bias: Optional[float] = None,
        token_shuffling: bool = False,
    ):
        super().__init__()
        assert 0 < keep_rate <= 1, "The keep_rate must be in (0,1]"

        self.bias = bias
        self.keep_rate = keep_rate
        self.token_shuffling = token_shuffling

    def forward(self, x: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """Drop tokens from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_sz, seq_len, dim)``.
        force_drop : bool, optional, default=False
            If True, the tokens are always dropped, even when the model is in
            evaluation mode.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_sz, keep_len, dim)`` containing the kept tokens.
        """
        if (not self.training and not force_drop) or self.keep_rate == 1:
            return x

        batch_sz, _, dim = x.shape

        cls_mask = torch.zeros(  # assumes that CLS is always the 1st element
            batch_sz, 1, dtype=torch.int64, device=x.device
        )
        patch_mask = self.uniform_mask(x)
        patch_mask = torch.hstack([cls_mask, patch_mask])

        return torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, dim))

    def uniform_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate token ids to keep from uniform random distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_sz, seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_sz, keep_len)`` containing the token ids to keep.

        """
        batch_sz, seq_len, _ = x.shape
        seq_len = seq_len - 1  # patch length (without CLS)

        keep_len = int(seq_len * self.keep_rate)
        noise = torch.rand(batch_sz, seq_len, device=x.device)
        if self.bias is not None:
            noise += self.bias
        ids = torch.argsort(noise, dim=1)
        keep_ids = ids[:, :keep_len]
        if not self.token_shuffling:
            keep_ids = keep_ids.sort(1)[0]
        return keep_ids
