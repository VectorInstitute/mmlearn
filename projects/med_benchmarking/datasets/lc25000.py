import os
from typing import Callable, Dict, Literal, Optional

import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from datasets import load_from_disk
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("LC25000_LUNG_ROOT_DIR", MISSING))
class LC25000(Dataset[Example]):
    """LC25000 dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory.
    organ : {'lung', 'colon'}, default='lung'
        Organ type ('lung' or 'colon').
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"],
        organ: Literal["lung", "colon"] = "lung",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the LC25000 dataset."""
        self.organ = organ
        dataset_path = os.path.join(root_dir, f"cache/lc25000_{organ}_{split}.arrow")

        if os.path.exists(dataset_path):
            print("!!! Using cached dataset")
            dataset = load_from_disk(dataset_path)
        else:
            raise ValueError("Dataset does not exist")

        self.transform = (
            Compose([Resize(224), CenterCrop(224), ToTensor()])
            if transform is None
            else transform
        )
        self.data = dataset

    @property
    def name(self) -> str:
        """Return the dataset name based on the organ (lung or colon)."""
        if self.organ == "lung":
            return "LC25000_lung"
        return "LC25000_colon"

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the label mapping."""
        if self.organ == "lung":
            return {
                0: "benign lung",
                1: "lung adenocarcinoma",
                2: "lung squamous cell carcinoma",
            }

        return {0: "benign colonic tissue", 1: "colon adenocarcinoma"}

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}.",
            "histopathology image of {}.",
            "pathology tissue showing {}.",
            "presence of {} tissue on image.",
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: int(entry["label"]),
                EXAMPLE_INDEX_KEY: idx,
            }
        )
