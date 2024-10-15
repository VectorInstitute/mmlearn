"""Tokenizers - modules that convert raw input to sequences of tokens."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from hydra_zen import store
from torch import nn
from transformers import AutoTokenizer

from mmlearn.datasets.core import Modalities


@store(group="datasets/tokenizers", provider="mmlearn")
class HFTokenizer:
    """HuggingFace tokenizer wrapper.

    This class wraps any huggingface tokenizer that can be initialized with
    `AutoTokenizer.from_pretrained`. It preprocesses the input text and returns
    the tokenized output.

    Parameters
    ----------
    model_name_or_path : str
        Pretrained model name or path - same as in `AutoTokenizer.from_pretrained`.
    max_length : int, optional, default=None
        Maximum length of the tokenized sequence. This is passed to the tokenizer
        `__call__` method.
    padding : bool or str, default=False
        Padding strategy. Same as in `AutoTokenizer.from_pretrained`; passed to
        the tokenizer `__call__` method.
    truncation : bool or str, optional, default=None
        Truncation strategy. Same as in `AutoTokenizer.from_pretrained`; passed to
        the tokenizer `__call__` method.
    **kwargs : Any
        Additional arguments passed to `AutoTokenizer.from_pretrained`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Optional[Union[bool, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(
        self, sentence: Union[str, List[str]], **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a text or a list of texts using the HuggingFace tokenizer.

        Parameters
        ----------
        sentence : str or list of str
            Sentence(s) to be tokenized.
        **kwargs : Any
            Additional arguments passed to the tokenizer `__call__` method.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized sentence(s).

        Notes
        -----
        The 'input_ids' key is replaced with 'Modalities.TEXT' for consistency.
        """
        batch_encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
            **kwargs,
        )

        if isinstance(
            sentence, str
        ):  # remove batch dimension if input is a single sentence
            for key, value in batch_encoding.items():
                if isinstance(value, torch.Tensor):
                    batch_encoding[key] = torch.squeeze(value, 0)

        # use 'Modalities.TEXT' key for input_ids for consistency
        batch_encoding[Modalities.TEXT] = batch_encoding["input_ids"]
        return dict(batch_encoding)


store(
    HFTokenizer,
    name="HFCLIPTokenizer",
    group="datasets/tokenizers",
    model_name_or_path="openai/clip-vit-base-patch16",
    padding=True,
    truncation=True,
)


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
