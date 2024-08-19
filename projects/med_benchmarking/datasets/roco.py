"""ROCO Dataset."""

import json
import os
from typing import Callable, Dict, Literal, Optional, Union

import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("ROCO_ROOT_DIR", MISSING))
class ROCO(Dataset[Example]):
    """ROCO dataset.

    Parameters
    ----------
    root_dir : str
        Path to the json file containing all entries of the dataset.
    split : {"train", "validation", "test"}
        Dataset split.
    group : {"radiology", "non-radiology"}, default="radiology"
        Dataset group.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    processor : Optional[Callable], default=None
        Function applied to the image-target pair.

    Notes
    -----
    If `processor` is not None, it overrides `transform` and `tokenizer`.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "validation", "test"] = "train",
        group: Literal["radiology", "non-radiology"] = "radiology",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        processor: Optional[
            Callable[[Image.Image, str], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        data_path = os.path.join(root_dir, group + split + "_dataset.json")
        with open(data_path, encoding="utf-8") as file:
            entries = [json.loads(line) for line in file.readlines()]
        self.entries = entries

        if processor is None and transform is None:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])
        elif processor is None:
            self.transform = transform
        else:
            self.transform = None

        if processor is None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None

        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample.

        If a tokenizer is not defined by `processor` or `tokenizer`, only the
        image and free text caption are returned. Otherwise, the image, free-
        text caption, and caption tokens are returned.
        """
        entry = self.entries[idx]
        with Image.open(entry["image_path"]) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption = entry["caption"]
        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        if self.processor is not None:
            image, tokens = self.processor(image, caption)

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: caption,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in tokens
                ), f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)
