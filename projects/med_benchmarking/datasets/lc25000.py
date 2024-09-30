import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from typing import Callable, Optional, Union, Dict
import torch

from omegaconf import MISSING
from mmlearn.conf import external_store
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES
from datasets import load_from_disk

@external_store(group="datasets", root_dir=os.getenv("LC25000_ROOT_DIR", MISSING))
class LC25000(Dataset[Example]):
    """LC25000 dataset for zero-shot classification.
    
    Parameters
    ----------
    root_dir : str
        Path to the dataset directory.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    processor : Optional[Callable], default=None
        Optional function to further process data samples.
    organ : str, default='lung'
        Organ type ('lung' or 'colon').
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
        organ: str = "lung",
    ) -> None:
        """Initialize the LC25000 dataset."""
        self.organ = organ
        organ_idx = 0 if organ == "lung" else 1
        dataset_path = os.path.join(root_dir, f"cache/lc25000_{organ}_{split}.arrow")

        if os.path.exists(dataset_path):
            print("!!! Using cached dataset")
            dataset = load_from_disk(dataset_path)
        else:
            raise ValueError(f"Dataset does not exist")

        self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()]) if transform is None else transform
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = dataset

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]
        label = entry["label"]
        label = self.get_label_mapping()[label]
        description = random.choice(TEMPLATES[self.name()])(label)
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        if self.transform is not None:
            image = self.transform(image)

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

    def name(self):
        if self.organ == "lung":
            return "LC25000_lung"
        elif self.organ == "colon":
            return "LC25000_colon"
        else:
            raise ValueError(f"Unknown organ: {self.organ}")
    
    def get_label_mapping(self):
        """Return label mapping based on the organ (lung or colon)."""
        if self.organ == "lung":
            return {
                0: "lung adenocarcinoma",
                1: "lung squamous cell carcinoma",
                2: "lung benign tissue",
            }
        elif self.organ == "colon":
            return {
                0: "colon adenocarcinoma",
                1: "colon benign tissue",
            }
        else:
            raise ValueError(f"Unknown organ: {self.organ}")

