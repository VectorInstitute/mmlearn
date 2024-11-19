"""Token mask generators."""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from hydra_zen import store
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@store(group="datasets/masking", provider="mmlearn", probability=0.15)
class RandomMaskGenerator:
    """Random mask generator.

    Returns a random mask of shape `(nb_patches, nb_patches)` based on the
    configuration where the number of patches to be masked is num_masking_patches.

    Parameters
    ----------
    probability : float
        Probability of masking a token.
    """

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(
        self,
        inputs: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a random mask.

        Returns a random mask of shape (nb_patches, nb_patches) based on the
        configuration where the number of patches to be masked is num_masking_patches.

        Returns
        -------
        inputs : torch.Tensor
            The encoded inputs.
        tokenizer : PreTrainedTokenizer
            The tokenizer.
        special_tokens_mask : Optional[torch.Tensor], default=None
            Mask for special tokens.
        """
        inputs = tokenizer.pad(inputs, return_tensors="pt")["input_ids"]
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        # (with probability `self.probability`)
        probability_matrix = torch.full(labels.shape, self.probability)
        if special_tokens_mask is None:
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                labels, already_has_special_tokens=True
            )
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = tokenizer.pad_token_id
        # 80% of the time, replace masked input tokens with tokenizer.mask_token([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # Rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices


@store(group="datasets/masking", provider="mmlearn")
class BlockwiseImagePatchMaskGenerator:
    """Blockwise image patch mask generator.

    Parameters
    ----------
    input_size: int or Tuple[int, int]
        The size of the input image.
    num_masking_patches: int
        The number of patches to be masked.
    min_num_patches: int, default=4
        The minimum number of patches to be masked.
    max_num_patches: int, default=None
        The maximum number of patches to be masked.
    min_aspect_ratio: float, default=0.3
        The minimum aspect ratio of the patch.
    max_aspect_ratio: float, default=None
        The maximum aspect ratio of the patch.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        num_masking_patches: int,
        min_num_patches: int = 4,
        max_num_patches: Any = None,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: Any = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect_ratio = max_aspect_ratio or 1 / min_aspect_ratio
        self.log_aspect_ratio = (math.log(min_aspect_ratio), math.log(max_aspect_ratio))

    def __repr__(self) -> str:
        """Generate a printable representation.

        Returns
        -------
        str
            A printable representation of the object.

        """
        return "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )

    def get_shape(self) -> Tuple[int, int]:
        """Get the shape of the mask."""
        return self.height, self.width

    def _mask(self, mask: torch.Tensor, max_mask_patches: int) -> int:
        """Masking function.

        This function mask adjacent patches by first selecting a target area and aspect
        ratio. Since, there might be overlap between selected areas  or the selected
        area might already be masked, it runs for a  maximum of 10 attempts or until the
        specified number of patches (max_mask_patches) is achieved.


        Parameters
        ----------
        mask: torch.Tensor
            Current mask. The mask to be updated.
        max_mask_patches: int
            The maximum number of patches to be masked.

        Returns
        -------
        delta: int
            The number of patches that were successfully masked.

        Notes
        -----
        - `target_area`: Randomly chosen target area for the patch.
        - `aspect_ratio`: Randomly chosen aspect ratio for the patch.
        - `h`: Height of the patch based on the target area and aspect ratio.
        - `w`: Width of the patch based on the target area and aspect ratio.
        - `top`: Randomly chosen top position for the patch.
        - `left`: Randomly chosen left position for the patch.
        - `num_masked`: Number of masked pixels within the proposed patch area.
        - `delta`: Accumulated count of modified pixels.
        """
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self) -> torch.Tensor:
        """Generate a random mask.

        Returns a random mask of shape (nb_patches, nb_patches) based on the
        configuration where the number of patches to be masked is num_masking_patches.

        Returns
        -------
        mask: torch.Tensor
            A mask of shape (nb_patches, nb_patches)

        """
        mask = torch.zeros(self.get_shape(), dtype=torch.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta

        return mask


def apply_masks(
    x: torch.Tensor, masks: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Apply masks to the input tensor by selecting the patches to keep based on the masks.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, N, D).
    masks : Union[torch.Tensor, List[torch.Tensor]]
        A list of mask tensors of shape (N,), (1, N), or (B, N).

    Returns
    -------
    torch.Tensor
        The masked tensor where only the patches indicated by the masks are kept.
        The output tensor has shape (B * num_masks, N', D), where N' is the number of patches kept.
    """
    all_x = []
    batch_size = x.size(0)
    for m in masks:
        m = m.to(x.device)

        # Ensure mask is at least 2D
        if m.dim() == 1:
            m = m.unsqueeze(0)  # Shape: (1, N)

        # Expand mask to match the batch size if needed
        if m.size(0) == 1 and batch_size > 1:
            m = m.expand(batch_size, -1)  # Shape: (B, N)

        # Expand mask to match x's dimensions
        m_expanded = (
            m.unsqueeze(-1).expand(-1, -1, x.size(-1)).bool()
        )  # Shape: (B, N, D)

        # Use boolean indexing
        selected_patches = x[m_expanded].view(batch_size, -1, x.size(-1))
        all_x.append(selected_patches)

    # Concatenate along the batch dimension
    return torch.cat(all_x, dim=0)


@dataclass
class IJEPAMaskGenerator:
    """Generates encoder and predictor masks for preprocessing.

    This class generates masks dynamically for batches of examples.

    Parameters
    ----------
    input_size : tuple[int, int], default=(224, 224)
        Input image size.
    patch_size : int, default=16
        Size of each patch.
    min_keep : int, default=4
        Minimum number of patches to keep.
    allow_overlap : bool, default=False
        Whether to allow overlap between encoder and predictor masks.
    enc_mask_scale : tuple[float, float], default=(0.2, 0.8)
        Scale range for encoder mask.
    pred_mask_scale : tuple[float, float], default=(0.2, 0.8)
        Scale range for predictor mask.
    aspect_ratio : tuple[float, float], default=(0.3, 3.0)
        Aspect ratio range for mask blocks.
    nenc : int, default=1
        Number of encoder masks to generate.
    npred : int, default=2
        Number of predictor masks to generate.
    """

    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    min_keep: int = 4
    allow_overlap: bool = False
    enc_mask_scale: Tuple[float, float] = (0.2, 0.8)
    pred_mask_scale: Tuple[float, float] = (0.2, 0.8)
    aspect_ratio: Tuple[float, float] = (0.3, 3.0)
    nenc: int = 1
    npred: int = 2

    def __post_init__(self) -> None:
        """Initialize the mask generator."""
        self.height = self.input_size[0] // self.patch_size
        self.width = self.input_size[1] // self.patch_size

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
    ) -> Tuple[int, int]:
        """Sample the size of the mask block based on scale and aspect ratio."""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        min_ar, max_ar = aspect_ratio
        aspect_ratio_val = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * aspect_ratio_val)))
        w = int(round(math.sqrt(max_keep / aspect_ratio_val)))

        h = min(h, self.height - 1)
        w = min(w, self.width - 1)

        return h, w

    def _sample_block_mask(
        self,
        b_size: Tuple[int, int],
        acceptable_regions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a mask block."""
        h, w = b_size
        top = torch.randint(0, self.height - h, (1,)).item()
        left = torch.randint(0, self.width - w, (1,)).item()
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[top : top + h, left : left + w] = 1

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0

        return mask.flatten(), mask_complement.flatten()

    def __call__(
        self,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Generate encoder and predictor masks for a batch of examples.

        Parameters
        ----------
        batch_size : int, default=1
            The batch size for which to generate masks.

        Returns
        -------
        Dict[str, Any]
            A dictionary of encoder masks and predictor masks.
        """
        seed = torch.randint(
            0, 2**32, (1,)
        ).item()  # Sample random seed for reproducibility
        g = torch.Generator().manual_seed(seed)

        # Sample block sizes
        p_size = self._sample_block_size(
            generator=g, scale=self.pred_mask_scale, aspect_ratio=self.aspect_ratio
        )
        e_size = self._sample_block_size(
            generator=g, scale=self.enc_mask_scale, aspect_ratio=(1.0, 1.0)
        )

        # Generate predictor masks
        masks_pred, masks_enc = [], []
        for _ in range(self.npred):
            mask_p, _ = self._sample_block_mask(p_size)
            # Expand mask to match batch size
            mask_p = mask_p.unsqueeze(0).expand(batch_size, -1)
            masks_pred.append(mask_p)

        # Generate encoder masks
        for _ in range(self.nenc):
            mask_e, _ = self._sample_block_mask(e_size)
            # Expand mask to match batch size
            mask_e = mask_e.unsqueeze(0).expand(batch_size, -1)
            masks_enc.append(mask_e)

        return {
            "encoder_masks": masks_enc,  # List of tensors of shape (batch_size, N)
            "predictor_masks": masks_pred,  # List of tensors of shape (batch_size, N)
        }
