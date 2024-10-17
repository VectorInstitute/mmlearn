"""Sicap Dataset."""

import os
from typing import Callable, Dict, Literal, Optional

import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("SICAP_ROOT_DIR", MISSING))
class SICAP(Dataset[Example]):
    """SICAP dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    split : {'train', 'test'}
        Dataset split, must be one of ["train", "test"].
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"],
        image_dir: str = "images",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the dataset."""
        assert split in ["train", "test"], f"split {split} is not supported in dataset."
        image_dir = os.path.join(root_dir, image_dir)

        if split == "train":
            csv_file = os.path.join(root_dir, "partition/Test", "Train.xlsx")
            self.data = pd.read_excel(csv_file)
        elif split == "test":
            csv_file = os.path.join(root_dir, "partition/Test", "Test.xlsx")
            self.data = pd.read_excel(csv_file)

        # Drop all columns except image_name and label columns
        label_columns = ["NC", "G3", "G4", "G5"]
        self.data = self.data[["image_name"] + label_columns]

        # Get the index of the maximum label value for each row
        self.data["labels"] = self.data[label_columns].idxmax(axis=1)

        # Replace label column values with categorical values
        self.cat_to_num_map = {
            "NC": 0,
            "G3": 1,
            "G4": 2,
            "G5": 3,
        }
        self.data["labels"] = self.data["labels"].map(self.cat_to_num_map)

        self.image_paths = self.data["image_name"].values
        self.labels = self.data["labels"].values
        self.image_dir = image_dir
        self.transform = (
            transform
            if transform is not None
            else Compose([Resize(224), CenterCrop(224), ToTensor()])
        )

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the label mapping."""
        return {
            0: "benign glands",
            1: "atrophic dense glands",
            2: "cribriform ill-formed fused papillary patterns",
            3: "isolated nest cells without lumen roseting patterns",
        }

    @property
    def zero_shot_prompt_templates(self) -> list[Callable[[str], str]]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}.",
            "histopathology image of {}.",
            "pathology tissue showing {}.",
            "presence of {} tissue on image.",
        ]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        label_index = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label_index,
                EXAMPLE_INDEX_KEY: idx,
            }
        )
