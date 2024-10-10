"""Test functionality of different transform module."""

import numpy as np
import torch

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.masking import RandomMaskGenerator
from mmlearn.datasets.processors.tokenizers import HFTokenizer


torch.manual_seed(42)
np.random.seed(42)


def test_random_masking() -> None:
    """Test random mask generator."""
    mask_probability = 0.5
    checkpoint_name = "roberta-base"
    mask_generator = RandomMaskGenerator(mask_probability)
    tokenizer = HFTokenizer(checkpoint_name, max_length=512, padding="max_length")
    tokenizer_output = tokenizer(
        "A photo of a cat, a quick brown fox jumps over the lazy dog!"
    )

    torch.testing.assert_close(
        tokenizer_output[Modalities.TEXT.name][:4], torch.tensor([0, 250, 1345, 9])
    )

    _, _, mask = mask_generator(
        tokenizer_output,
        tokenizer.tokenizer,  # type: ignore
    )

    assert len(mask) == 512
    assert sum(mask) != 0, "Mask should not be empty"
