"""Vision Transformer encoder for data2vec."""

from typing import Any, Dict, Tuple, Union

import torch
from hydra_zen import store
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality


@store(group="modules/encoders", provider="mmlearn", hydra_convert="object")
class Data2VecVisionEncoder(nn.Module):
    """Vision Transformer encoder for data2vec.

    Parameters
    ----------
    model_name : str, default='vit_base_patch16_224'
        The name of the Vision Transformer model to use from timm.
    pretrained : bool, default=True
        Whether to use pretrained weights.
    num_classes : int, default=0
        The number of classes for the classification head.
        Set to 0 to remove the classification head.
    drop_path_rate : float, default=0.1
        The drop path rate for the model.
    mask_ratio : float, default=0.6
        The ratio of patches to mask during training.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 0,
        drop_path_rate: float = 0.1,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, inputs: Dict[Union[str, Modality], Any]
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The image tensor will be expected under the
            `Modalities.IMAGE` key.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing:
            - 'last_hidden_state': The output of the last layer of the encoder.
            - 'hidden_states': The outputs of all layers of the encoder.
            - 'mask': The mask applied to the input patches.
            - 'cls_token': The class token output (if present in the model).

        Raises
        ------
        ValueError
            If the input image dimensions are not 224x224.
        """
        x = inputs[Modalities.IMAGE]
        batch_size, channels, height, width = x.shape
        if height != 224 or width != 224:
            raise ValueError(f"Input image should be 224x224, got {height}x{width}")

        # Create patch embedding
        x = self.model.patch_embed(x)

        # Add position embedding
        cls_token = self.model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed

        # Apply masking
        num_patches = x.shape[1] - 1  # exclude cls token
        num_mask = int(self.mask_ratio * num_patches)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=x.device)
        mask[:, torch.randperm(num_patches)[:num_mask]] = True

        x_masked = x.clone()
        mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)
        x_masked[:, 1:, :] = torch.where(mask.unsqueeze(-1), mask_tokens, x[:, 1:, :])

        # Apply transformer blocks
        hidden_states = []
        for blk in self.model.blocks:
            x_masked = blk(x_masked)
            hidden_states.append(x_masked)

        # Apply layer norm
        x_masked = self.model.norm(x_masked)

        return {
            "last_hidden_state": x_masked[:, 1:],  # exclude cls token
            "hidden_states": [hs[:, 1:] for hs in hidden_states],  # exclude cls token
            "mask": mask,
            "cls_token": x_masked[:, 0],
        }

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Get the output of the intermediate layers.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        n : int, default=1
            The number of intermediate layers to return.

        Returns
        -------
        torch.Tensor
            The output of the intermediate layers.
        """
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed

        # Apply transformer blocks
        for i, blk in enumerate(self.model.blocks):
            if i == len(self.model.blocks) - n:
                return self.model.norm(x)
            x = blk(x)

        return self.model.norm(x)

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
