"""Sicap Dataset."""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from typing import Callable, Optional, Union, Dict
import torch
import random
from omegaconf import MISSING

from mmlearn.conf import external_store
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES

@external_store(group="datasets", root_dir=os.getenv("SICAP_ROOT_DIR", MISSING))
class SICAP(Dataset[Example]):
    """SICAP dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        image_dir: str = "images",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        processor: Optional[
            Callable[[torch.Tensor, str], tuple[torch.Tensor, str]]
        ] = None,
        split: str = "test",
    ) -> None:
        """Initialize the dataset."""
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
        self.transform = transform if transform is not None else Compose([Resize(224), CenterCrop(224), ToTensor()])
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        label_index = self.labels[idx]
        label = list(self.cat_to_num_map.keys())[label_index]
        description = random.choice(TEMPLATES[self.__clas__.__name__])(label)
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # Process image and tokens if needed
        if self.processor is not None:
            image, tokens = self.processor(image, label)

        # Create an Example instance
        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: label,
                Modalities.RGB.target: label_index,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        # Add tokens to the example if available
        if tokens is not None:
            if isinstance(tokens, dict):  # If using a Hugging Face tokenizer
                assert Modalities.TEXT in tokens, f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def get_label_mapping(self):
        """Return the label mapping."""
        return {
            "NC": "benign glands",
            "G3": "atrophic dense glands",
            "G4": "cribriform ill-formed fused papillary patterns",
            "G5": "isolated nest cells without lumen roseting patterns",
        }

