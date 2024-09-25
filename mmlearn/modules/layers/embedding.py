"""Embedding layers utilities."""

from typing import List, Tuple

import numpy as np
import torch
from torch import nn


def patchify(batch: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """Patchify a batch of images.

    Parameters
    ----------
    batch : torch.Tensor
        Batch of images.
    patch_size : tuple of int
        The size of the patch.

    Returns
    -------
    torch.Tensor
        Patchified batch.

    Notes
    -----
    - Input shape: (b, h, w, c)
    - Output shape: (b, nh, nw, ph, pw, c)

    """
    b, c, h, w = batch.shape
    ph, pw = patch_size
    nh, nw = h // ph, w // pw

    batch_patches = torch.reshape(batch, (b, c, nh, ph, nw, pw))
    return torch.permute(batch_patches, (0, 1, 2, 4, 3, 5))


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module divides an image into patches and embeds them as a sequence of vectors.

    Parameters
    ----------
    img_size : int, optional, default=224
        Size of the input image (assumed to be square).
    patch_size : int, optional, default=16
        Size of each image patch (assumed to be square).
    in_chans : int, optional, default=3
        Number of input channels in the image.
    embed_dim : int, optional, default=768
        Dimension of the output embeddings.

    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to convert an image into patch embeddings."""
        return self.proj(x).flatten(2).transpose(1, 2)


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models.

    This module builds convolutional stems for Vision Transformers (ViT)
    with intermediate batch normalization and ReLU activation.

    Parameters
    ----------
    channels : List[int]
        List of channel sizes for each convolution layer.
    strides : List[int]
        List of stride sizes for each convolution layer.
    img_size : int, optional, default=224
        Size of the input image (assumed to be square).
    in_chans : int, optional, default=3
        Number of input channels in the image.
    batch_norm : bool, optional, default=True
        Whether to include batch normalization after each convolution layer.

    """

    def __init__(
        self,
        channels: List[int],
        strides: List[int],
        img_size: int = 224,
        in_chans: int = 3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=strides[i],
                    padding=1,
                    bias=(not batch_norm),
                )
            ]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i + 1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [
            nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])
        ]
        self.stem = nn.Sequential(*stem)

        # Compute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size // stride_prod) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional embedding layers."""
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class Img2Seq(nn.Module):
    """Convert a batch of images to a batch of sequences.

    Parameters
    ----------
    img_size : tuple of int
        The size of the input image.
    patch_size : tuple of int
        The size of the patch.
    n_channels : int
        The number of channels in the input image.
    d_model : int
        The dimension of the output sequence.

    Notes
    -----
    - Input shape: (b, h, w, c)
    - Output shape: (b, s, d)

    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        n_channels: int,
        d_model: int,
    ) -> None:
        """Initialize the Img2Seq module."""
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = nh * nw

        token_dim = patch_size[0] * patch_size[1] * n_channels
        self.linear = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model))

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to a batch of sequences."""
        batch = patchify(batch, self.patch_size)

        b, c, nh, nw, ph, pw = batch.shape

        # Flattening the patches
        batch = torch.permute(batch, [0, 2, 3, 4, 5, 1])
        batch = torch.reshape(batch, [b, nh * nw, ph * pw * c])

        batch = self.linear(batch)
        cls: torch.Tensor = self.cls_token.expand([b, -1, -1])
        emb: torch.Tensor = batch + self.pos_emb
        return torch.cat([cls, emb], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
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

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
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
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)
