"""Vision encoder implementations."""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import timm
import torch
from hydra_zen import store
from peft import PeftConfig
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core.modalities import Modalities
from mmlearn.datasets.processors.masking import apply_masks
from mmlearn.datasets.processors.transforms import (
    repeat_interleave_batch,
    trunc_normal_,
)
from mmlearn.modules.layers.embedding import PatchEmbed, get_2d_sincos_pos_embed
from mmlearn.modules.layers.transformer_block import Block


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name="vit_base_patch16_224",
    hydra_convert="object",
)
class TimmViT(nn.Module):
    """Vision Transformer model from timm.

    Parameters
    ----------
    model_name : str
        The name of the model to use.
    modality : str, default="RGB"
        The modality of the input data. This allows this model to be used with different
        image modalities e.g. RGB, Depth, etc.
    projection_dim : int, default=768
        The dimension of the projection head.
    pretrained : bool, default=True
        Whether to use the pretrained weights.
    freeze_layers : Union[int, float, List[int], bool], default=False
        Whether to freeze the layers.
    freeze_layer_norm : bool, default=True
        Whether to freeze the layer norm.
    peft_config : Optional[PeftConfig], default=None
        The PEFT configuration.
    model_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments for the model.
    """

    def __init__(
        self,
        model_name: str,
        modality: str = "RGB",
        projection_dim: int = 768,
        pretrained: bool = True,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional[PeftConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Vision Transformer model."""
        super().__init__()
        self.modality = Modalities.get_modality(modality)
        if model_kwargs is None:
            model_kwargs = {}

        self.model: TimmVisionTransformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=projection_dim,
            **model_kwargs,
        )
        assert isinstance(self.model, TimmVisionTransformer), (
            f"Model {model_name} is not a Vision Transformer. "
            "Please provide a model name that corresponds to a Vision Transformer."
        )

        self._freeze_layers(freeze_layers, freeze_layer_norm)

        if peft_config is not None:
            self.model = hf_utils._wrap_peft_model(self.model, peft_config)

    def _freeze_layers(
        self, freeze_layers: Union[int, float, List[int], bool], freeze_layer_norm: bool
    ) -> None:
        """Freeze the layers of the model.

        Parameters
        ----------
        freeze_layers : Union[int, float, List[int], bool]
            Whether to freeze the layers.
        freeze_layer_norm : bool
            Whether to freeze the layer norm.
        """
        if isinstance(freeze_layers, bool) and freeze_layers:
            for name, param in self.model.named_parameters():
                param.requires_grad = (
                    (not freeze_layer_norm) if "norm" in name else False
                )

        modules = [self.model.patch_embed, *self.model.blocks, self.model.norm]
        if isinstance(freeze_layers, float):
            freeze_layers = int(freeze_layers * len(modules))
        if isinstance(freeze_layers, int):
            freeze_layers = list(range(freeze_layers))

        if isinstance(freeze_layers, list):
            for idx, module in enumerate(modules):
                if idx in freeze_layers:
                    for name, param in module.named_parameters():
                        param.requires_grad = (
                            (not freeze_layer_norm) if "norm" in name else False
                        )

    def forward(self, inputs: Dict[str, Any]) -> BaseModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input data. The `image` will be expected under the `Modalities.RGB` key.

        Returns
        -------
        BaseModelOutput
            The output of the model.
        """
        x = inputs[self.modality.name]
        last_hidden_state, hidden_states = self.model.forward_intermediates(
            x, output_fmt="NLC"
        )
        last_hidden_state = self.model.forward_head(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )

    def get_intermediate_layers(
        self, inputs: Dict[str, Any], n: int = 1
    ) -> List[torch.Tensor]:
        """Get the output of the intermediate layers.

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input data. The `image` will be expected under the `Modalities.RGB` key.
        n : int, default=1
            The number of intermediate layers to return.

        Returns
        -------
        List[torch.Tensor]
            The outputs of the last n intermediate layers.
        """
        return self.model.get_intermediate_layers(inputs[Modalities.RGB], n)  # type: ignore

    def get_patch_info(self) -> Tuple[int, int]:
        """Get patch size and number of patches.

        Returns
        -------
        Tuple[int, int]
            Patch size and number of patches.
        """
        patch_size = self.model.patch_embed.patch_size[0]
        num_patches = self.model.patch_embed.num_patches
        return patch_size, num_patches


class VisionTransformer(nn.Module):
    """Vision Transformer.

    This module implements a Vision Transformer that processes images using a
    series of transformer blocks and patch embeddings.

    Parameters
    ----------
    modality : str, optional, default="RGB"
        The modality of the input data. This allows this model to be used with different
        image modalities e.g. RGB, Depth, etc.
    img_size : List[int], optional, default=None
        List of input image sizes.
    patch_size : int, optional, default=16
        Size of each patch.
    in_chans : int, optional, default=3
        Number of input channels.
    embed_dim : int, optional, default=768
        Embedding dimension.
    predictor_embed_dim : int, optional, default=384
        Embedding dimension for the predictor.
    depth : int, optional, default=12
        Number of transformer blocks.
    predictor_depth : int, optional, default=12
        Number of transformer blocks in the predictor.
    num_heads : int, optional, default=12
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of hidden dimension in the MLP.
    qkv_bias : bool, optional, default=True
        If True, add a learnable bias to the query, key, and value projections.
    qk_scale : Optional[float], optional
        Override the default qk scale factor.
    drop_rate : float, optional, default=0.0
        Dropout rate for the transformer blocks.
    attn_drop_rate : float, optional, default=0.0
        Dropout rate for the attention mechanism.
    drop_path_rate : float, optional, default=0.0
        Dropout rate for stochastic depth.
    norm_layer : Callable[..., nn.Module], optional, default=nn.LayerNorm
        Normalization layer to use.
    init_std : float, optional, default=0.02
        Standard deviation for weight initialization.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        modality: str = "RGB",
        img_size: Optional[List[int]] = None,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        predictor_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vision Transformer module."""
        super().__init__()
        self.modality = Modalities.get_modality(modality)
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        img_size = [224, 224] if img_size is None else img_size

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Transformer Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Weight Initialization
        self.init_std = init_std
        self.apply(self._init_weights)

    def fix_init_weight(self) -> None:
        """Fix initialization of weights by rescaling them according to layer depth."""

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp[-1].weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for the layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, inputs: Dict[str, Any], return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass through the Vision Transformer."""
        masks = inputs.get(self.modality.mask)
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = inputs[self.modality.name]
        # -- Patchify x
        x = self.patch_embed(x)

        # -- Add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- Mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- Initialize a list to store hidden states
        hidden_states: Optional[List[torch.Tensor]] = (
            [] if return_hidden_states else None
        )

        # -- Forward propagation through blocks
        for _i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_hidden_states and hidden_states is not None:
                hidden_states.append(x)

        # -- Apply normalization if present
        if self.norm is not None:
            x = self.norm(x)

        # -- Return both final output and hidden states if requested
        if return_hidden_states:
            return x, hidden_states
        return (x, None)

    def interpolate_pos_encoding(
        self, x: torch.Tensor, pos_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate positional encoding to match the size of the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        pos_embed : torch.Tensor
            Positional embedding tensor.

        Returns
        -------
        torch.Tensor
            Interpolated positional encoding.
        """
        npatch = x.shape[1] - 1
        n = pos_embed.shape[1] - 1
        if npatch == n:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(n)), int(math.sqrt(n)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / n),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer Predictor.

    This module implements a Vision Transformer that predicts masked tokens
    using a series of transformer blocks.

    Parameters
    ----------
    num_patches : int
        The number of patches in the input image.
    embed_dim : int, optional, default=768
        The embedding dimension.
    predictor_embed_dim : int, optional, default=384
        The embedding dimension for the predictor.
    depth : int, optional, default=6
        The number of transformer blocks.
    num_heads : int, optional, default=12
        The number of attention heads.
    mlp_ratio : float, optional, default=4.0
        Ratio of the hidden dimension in the MLP.
    qkv_bias : bool, optional, default=True
        If True, add a learnable bias to the query, key, and value projections.
    qk_scale : Optional[float], optional, default=None
        Override the default qk scale factor.
    drop_rate : float, optional, default=0.0
        Dropout rate for the transformer blocks.
    attn_drop_rate : float, optional, default=0.0
        Dropout rate for the attention mechanism.
    drop_path_rate : float, optional, default=0.0
        Dropout rate for stochastic depth.
    norm_layer : Callable[..., nn.Module], optional, default=nn.LayerNorm
        Normalization layer to use.
    init_std : float, optional, default=0.02
        Standard deviation for weight initialization.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        num_patches: int = 196,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vision Transformer Predictor module."""
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.predictor_embed = nn.Linear(self.embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # Positional Embedding
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim), requires_grad=False
        )
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=False,
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )

        # Transformer Blocks
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Weight Initialization
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)

    def fix_init_weight(self) -> None:
        """Fix initialization of weights by rescaling them according to layer depth."""

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for the layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass through the Vision Transformer Predictor."""
        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        b = len(x) // len(masks_x)

        # -- Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # -- Add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(b, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, n_ctxt, d = x.shape

        # -- Concatenate mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(b, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, b, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- Forward propagation
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- Return predictions for mask tokens
        x = x[:, n_ctxt:]
        return self.predictor_proj(x)


@cast(
    VisionTransformerPredictor,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_predictor(**kwargs: Any) -> VisionTransformerPredictor:
    """
    Create a VisionTransformerPredictor model.

    Returns
    -------
    VisionTransformerPredictor
        An instance of VisionTransformerPredictor.
    """
    return VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_tiny(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with tiny configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_small(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with small configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_base(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with base configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_large(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with large configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_huge(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with huge configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


@cast(
    VisionTransformer,
    store(
        group="modules/encoders",
        provider="mmlearn",
    ),
)
def vit_giant(patch_size: int = 16, **kwargs: Any) -> VisionTransformer:
    """
    Create a VisionTransformer model with giant configuration.

    Returns
    -------
    VisionTransformer
        An instance of VisionTransformer.
    """
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS: Dict[str, int] = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
}
