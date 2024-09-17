"""Vision encoder implementations."""

from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
from hydra_zen import store
from peft import PeftConfig
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core.modalities import Modalities, Modality


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name_or_path="vit_base_patch16_224",
    hydra_convert="object",
)
class TimmViT(nn.Module):
    """Vision Transformer model from timm.

    Parameters
    ----------
    model_name : str
        The name of the model to use.
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
        projection_dim: int = 768,
        pretrained: bool = True,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional[PeftConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Vision Transformer model."""
        super().__init__()
        if model_kwargs is None:
            model_kwargs = {}

        self.model: VisionTransformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=projection_dim,
            **model_kwargs,
        )
        assert isinstance(self.model, VisionTransformer), (
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

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> BaseModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The `image` will be expected under the `Modalities.RGB` key.

        Returns
        -------
        BaseModelOutput
            The output of the model.
        """
        x = inputs[Modalities.RGB]
        last_hidden_state, hidden_states = self.model.forward_intermediates(
            x, output_fmt="NLC"
        )
        last_hidden_state = self.model.forward_head(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )

    def get_intermediate_layers(
        self, inputs: Dict[Union[str, Modality], Any], n: int = 1
    ) -> List[torch.Tensor]:
        """Get the output of the intermediate layers.

        Parameters
        ----------
        inputs : Dict[Union[str, Modality], Any]
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
