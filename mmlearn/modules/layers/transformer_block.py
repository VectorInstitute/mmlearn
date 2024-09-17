"""Transformer Block and Embedding Modules for Vision Transformers (ViT)."""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

from mmlearn.modules.layers.attention import Attention
from mmlearn.modules.layers.mlp import MLP


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
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
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Parameters
    ----------
    drop_prob : Optional[float], optional
        Probability of dropping paths. Default is None.
    """

    def __init__(self, drop_prob: float) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DropPath module."""
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    """
    Transformer Block.

    This module represents a Transformer block that includes self-attention,
    normalization layers, and a feedforward multi-layer perceptron (MLP) network.

    Parameters
    ----------
    dim : int
        The input and output dimension of the block.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of hidden dimension to the input dimension in the MLP.
    qkv_bias : bool, optional, default=False
        If True, add a learnable bias to the query, key, value projections.
    qk_scale : float, optional, default=None
        Override default qk scale of head_dim ** -0.5 if set.
    drop : float, optional, default=0.0
        Dropout probability for the output of attention and MLP layers.
    attn_drop : float, optional, default=0.0
        Dropout probability for the attention scores.
    drop_path : float, optional, default=0.0
        Stochastic depth rate, a form of layer dropout.
    act_layer : Callable[..., nn.Module], optional, default=nn.GELU
        Activation layer in the MLP.
    norm_layer : Callable[..., nn.Module], optional, default=nn.LayerNorm
        Normalization layer.

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dims=[mlp_hidden_dim],
            activation_layer=act_layer,
            dropout=drop,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Forward pass through the Transformer Block."""
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x)))


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
