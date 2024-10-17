"""NCK CRC Dataset."""

import os
from typing import Callable, Dict, Literal, Optional

import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from datasets import load_dataset, load_from_disk
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("NCK_CRC_ROOT_DIR", MISSING))
class NckCrc(Dataset[Example]):
    """NCK CRC dataset for colorectal cancer classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str, default='train'
        Dataset split, one of 'train', 'train_nonorm', or 'validation'.
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "train_nonorm", "validation"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the NCK CRC dataset."""
        assert split in (
            "train",
            "train_nonorm",
            "validation",
        ), f"Invalid split: {split}"

        # Class mapping for labels
        self.class_mapping = {
            "ADI": 0,
            "DEB": 1,
            "LYM": 2,
            "MUC": 3,  # noqa: F821
            "MUS": 4,
            "NORM": 5,
            "STR": 6,
            "TUM": 7,
        }

        # Load cached dataset if it exists, otherwise download and cache it
        cache_path = os.path.join(root_dir, f"cache/nck_crc_{split}.arrow")
        if os.path.exists(cache_path):
            print(f"Using cached dataset: {cache_path}")
            dataset = load_from_disk(cache_path)
        else:
            os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)
            dataset = load_dataset(
                "DykeF/NCTCRCHE100K",
                cache_dir=os.path.join(root_dir, "scratch/"),
                split=split,
            )
            dataset = dataset.filter(
                lambda row: row["label"] != "BACK"
            )  # Exclude "BACK" label
            dataset.save_to_disk(cache_path)

        self.data = dataset

        self.transform = (
            Compose([Resize(224), CenterCrop(224), ToTensor()])
            if transform is None
            else transform
        )

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]
        label = self.class_mapping[entry["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the label mapping for the NCK CRC dataset."""
        return {
            0: "adipose",
            1: "debris",
            2: "lymphocytes",
            3: "mucus",
            4: "smooth muscle",
            5: "normal colon mucosa",
            6: "cancer-associated stroma",
            7: "colorectal adenocarcinoma epithelium",
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
