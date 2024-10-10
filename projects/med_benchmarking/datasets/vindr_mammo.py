"""VinDr-Mammo Dataset."""

import os
import pickle
import random
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("VINDR_MAMMO_ROOT_DIR", MISSING))
class VinDrMammo(Dataset[Example]):
    """VinDr-Mammo dataset for breast lesion classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata.
    split : str
        Dataset split, one of 'training' or 'test'.
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        processor: Optional[
            Callable[[torch.Tensor, str], tuple[torch.Tensor, str]]
        ] = None,
    ) -> None:
        """Initialize the VinDr-Mammo dataset."""
        assert split in ["training", "test"], f"split {split} is not supported."

        self.root_dir = root_dir
        self.split = split

        # Load cached dataset if available
        cache_path = f"cache/{split}_dataset.pkl"
        if os.path.exists(cache_path):
            print(f"Using cached dataset: {cache_path}")
            with open(cache_path, "rb") as f:
                self.entries = pickle.load(f)
        else:
            raise FileNotFoundError(f"Dataset cache not found at {cache_path}")

        # Only consider multi-class samples
        self.entries = [entry for entry in self.entries if sum(entry["label"]) < 2]
        for entry in self.entries:
            entry["label"] = np.argmax(entry["label"])

        if processor is None and transform is None:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])
        elif processor is None:
            self.transform = transform
        else:
            self.transform = None

        self.processor = processor
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.entries[idx]
        image_path = os.path.join(self.root_dir, entry["path"])
        label = entry["label"]
        label = self.get_label_mapping()[label]
        description = random.choice(
            [
                "a x-ray image showing {c}",
                "mammography image of {c}",
                "mammogram showing {c}",
                "presence of {c} on mammogram",
            ]
        )(label)
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        tokens = None
        if self.processor is not None:
            image, tokens = self.processor(image, label)

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: label,
                Modalities.RGB.target: int(entry["label"]),
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):
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

    def get_label_mapping(self):
        """Return the label mapping for the VinDr-Mammo dataset."""
        return {
            0: "Mass",
            1: "Suspicious Calcification",
            2: "Asymmetry",
            3: "Focal Asymmetry",
            4: "Global Asymmetry",
            5: "Architectural Distortion",
            6: "Skin Thickening",
            7: "Skin Retraction",
            8: "Nipple Retraction",
            9: "Suspicious Lymph Node",
        }
