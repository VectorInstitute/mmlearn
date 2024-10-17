"""PCAM Dataset."""

import os
import pickle
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


@external_store(group="datasets", root_dir=os.getenv("PCAM_ROOT_DIR", MISSING))
class PCAM(Dataset[Example]):
    """PCAM dataset for classification tasks.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str
        Dataset split (e.g., 'train', 'test', 'validation').
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    processor : Optional[Callable], default=None
        Function to process image and token pairs.
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
        """Initialize the PCAM dataset."""
        cache_path = os.path.join(root_dir, f"cache/pcam_{split}.pkl")

        if os.path.exists(cache_path):
            print("!!!Using cached dataset")
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)
            dataset = load_dataset(
                "1aurent/PatchCamelyon", cache_dir=os.path.join(root_dir, "scratch/")
            )[split]

            self.data = dataset
            with open(cache_path, "wb") as f:
                pickle.dump(self.data, f)

        self.transform = transform or Compose(
            [Resize(224), CenterCrop(224), ToTensor()]
        )
        self.tokenizer = tokenizer
        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"].convert("RGB")
        label_idx = int(entry["label"])

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label_idx,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    @property
    def label_mapping(self) -> Dict[str, str]:
        """Return the mapping of labels for the PCAM dataset."""
        return {0: "lymph node", 1: "lymph node containing metastatic tumor tissue"}

    @property
    def zero_shot_prompt_templates(self) -> list[Callable[[str], str]]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image",
        ]
