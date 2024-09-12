"""Attention modules for Vision Transformer (ViT) and related models."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embeddings.
    grid_size : int
        The size of the grid (both height and width).
    cls_token : bool, optional, default=False
        Whether to include a class token in the embeddings.

    Returns
    -------
    pos_embed : np.ndarray
        Positional embeddings with shape [grid_size*grid_size, embed_dim] or 
        [1 + grid_size*grid_size, embed_dim] if cls_token is True.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings from a grid.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embeddings.
    grid : np.ndarray
        The grid of positions with shape [2, 1, grid_size, grid_size].

    Returns
    -------
    emb : np.ndarray
        Positional embeddings with shape [grid_size*grid_size, embed_dim].
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate 1D sine-cosine positional embeddings.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embeddings.
    grid_size : int
        The size of the grid.
    cls_token : bool, optional, default=False
        Whether to include a class token in the embeddings.

    Returns
    -------
    pos_embed : np.ndarray
        Positional embeddings with shape [grid_size, embed_dim] or 
        [1 + grid_size, embed_dim] if cls_token is True.
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sine-cosine positional embeddings from a grid.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embeddings.
    pos : np.ndarray
        A list of positions to be encoded, with shape [M,].

    Returns
    -------
    emb : np.ndarray
        Positional embeddings with shape [M, embed_dim].
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) for regularization during training.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    drop_prob : float, optional, default=0.0
        Probability of dropping paths.
    training : bool, optional, default=False
        Whether the model is in training mode.

    Returns
    -------
    output : torch.Tensor
        Output tensor after applying drop path.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in the main path of residual blocks).

    Parameters
    ----------
    drop_prob : Optional[float], optional
        Probability of dropping paths. Default is None.
    """

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DropPath module."""
        return drop_path(x, self.drop_prob, self.training)


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
    qk_scale : Optional[float], optional
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
        proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-head self-attention module."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
