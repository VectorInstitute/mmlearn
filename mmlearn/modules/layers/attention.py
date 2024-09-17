"""Attention modules for Vision Transformer (ViT) and related models."""

from typing import Optional, Tuple

import torch
from torch import nn


class Attention(nn.Module):
    """
    Multi-head Self-Attention Mechanism.

    Parameters
    ----------
    dim : int
        Number of input dimensions.
    num_heads : int, optional, default=8
        Number of attention heads.
    qkv_bias : bool, optional, default=False
        If True, adds a learnable bias to the query, key, value projections.
    qk_scale : Optional[float], optional, default=None
        Override the default scale factor for the dot-product attention.
    attn_drop : float, optional, default=0.0
        Dropout probability for the attention weights.
    proj_drop : float, optional, default=0.0
        Dropout probability for the output of the attention layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-head self-attention module."""
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
