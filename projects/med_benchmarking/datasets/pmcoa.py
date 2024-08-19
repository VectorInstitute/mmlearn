"""PMC-OA dataset."""

import os
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import jsonlines
import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("PMCOA_ROOT_DIR", MISSING))
class PMCOA(Dataset[Example]):
    """Handles loading and processing of the PMC-OA dataset."""

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        file_type: str = "jsonl",
        image_key: str = "image",
        caption_key: str = "caption",
        csv_separator: str = ",",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[Callable[[str], torch.Tensor]] = None,
        mask_generator: Optional[
            Callable[
                [Dict[str, torch.Tensor], Any],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ] = None,
        image_dir: Optional[str] = None,
    ) -> None:
        """Initialize the dataset object with file paths and configurations.

        Parameters
        ----------
        root_dir : str
            Directory where the dataset is stored.
        split : str, default="train"
            Split of the dataset (train, valid, test).
        file_type : str, default="jsonl"
            Type of the input file (csv or jsonl).
        img_key : str, default="image"
            Key for images in the CSV/JSONL files.
        caption_key : str, default="caption"
            Key for captions in the CSV/JSONL files.
        csv_separator : str, default=","
            Separator used in CSV files. Not used for JSONL.
        transform : Callable, optional, default=None
            Transform applied to images.
        tokenizer : Callable[[torch.Tensor], Dict[str, torch.Tensor]]
            Text tokenizer.
        mask_generator : Callable[[Dict[str, torch.Tensor], Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], optional, default=None
            Generator for the mask.
        image_dir : str, optional, default=None
            Directory where images are stored, relative to the root directory.
            If not provided, it is assumed to be `'images'`.
        """  # noqa: W505
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "Invalid split name. Split must be one of 'train', 'valid', or 'test'."
            )
        if file_type not in ["csv", "jsonl"]:
            raise ValueError(
                "Invalid file type. File type must be one of 'csv' or 'jsonl'."
            )

        self.root_dir = root_dir

        if image_dir is None:
            self.image_dir = "images"
        else:
            self.image_dir = image_dir

        self.split = split
        input_filename = os.path.join(root_dir, f"{self.split}.{file_type}")

        self.image_filenames, self.captions = (
            self._csv_loader(input_filename, image_key, caption_key, csv_separator)
            if file_type == "csv"
            else self._jsonl_loader(input_filename, image_key, caption_key)
        )

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.tokenizer = tokenizer
        self.mask_generator = mask_generator

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.captions)

    def __getitem__(self, idx: int) -> Example:
        """Return items in the dataset."""
        image_path = os.path.join(
            self.root_dir, self.image_dir, self.image_filenames[idx]
        )

        with Image.open(image_path) as img:
            images = self.transform(img)

        caption = str(self.captions[idx])
        example = Example(
            {
                Modalities.RGB: images,
                Modalities.TEXT: caption,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None
        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in tokens
                ), f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        if self.mask_generator is not None and self.tokenizer is not None:
            _, masked_labels, masked_text = self.mask_generator(
                tokens,
                self.tokenizer.tokenizer,  # type: ignore
            )
            example[Modalities.TEXT.mask] = masked_text
            example[Modalities.TEXT.target] = masked_labels

        return example

    def _csv_loader(
        self, input_filename: str, img_key: str, caption_key: str, sep: str
    ) -> Tuple[Any, Any]:
        """Load images, captions from CSV data."""
        df = pd.read_csv(input_filename, sep=sep)
        images, captions = df[img_key].tolist(), df[caption_key].tolist()
        return images, captions

    def _jsonl_loader(
        self, input_filename: str, img_key: str, caption_key: str
    ) -> Tuple[Any, Any]:
        """Load images, captions from JSON data."""
        images, captions = [], []
        with jsonlines.open(input_filename) as reader:
            for obj in reader:
                images.append(obj[img_key])
                captions.append(obj[caption_key])
        return images, captions
