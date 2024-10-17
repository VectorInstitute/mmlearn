"""BACH Dataset."""

import os
import random
from typing import Callable, Dict, Optional, Union

import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from datasets import load_dataset
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("BACH_ROOT_DIR", MISSING))
class BACH(Dataset[Example]):
    """BACH dataset for breast cancer classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str
        Dataset split, one of 'train' or 'test'.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        processor: Optional[
            Callable[[torch.Tensor, str], tuple[torch.Tensor, str]]
        ] = None,
    ) -> None:
        """Initialize the BACH dataset."""
        os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)

        dataset = load_dataset(
            "1aurent/BACH",
            cache_dir=os.path.join(root_dir, "scratch/"),
            split="train",
        )
        data_dict = dataset.train_test_split(
            test_size=0.25, train_size=0.75, shuffle=True, seed=0
        )
        self.data = data_dict[split]

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
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]
        label = int(entry["label"])

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: int(entry["label"]),
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    @property
    def label_mapping(self) -> Dict[str, str]:
        """Return the label mapping for the BACH dataset."""
        return {
            0: "breast non-malignant benign tissue",
            1: "breast malignant in-situ carcinoma",
            2: "breast malignant invasive carcinoma",
            3: "breast normal breast tissue",
        }

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image",
        ]
