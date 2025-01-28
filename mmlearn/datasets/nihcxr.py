"""NIH Chest X-ray Dataset."""

import json
import os
from typing import Callable, Literal, Optional

import torch
from hydra_zen import MISSING, store
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


# NIH Chest X-ray disease labels
_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
]


@store(
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("NIH_CXR_DIR", MISSING),
    split="train",
)
class NIHCXR(Dataset[Example]):
    """NIH Chest X-ray dataset.

    Parameters
    ----------
    root_dir : str
        Directory which contains `.json` files stating all dataset entries.
    split : {"train", "test", "bbox"}
        Dataset split. "bbox" is a subset of "test" which contains bounding box info.
    transform : Optional[Callable[[PIL.Image], torch.Tensor]], optional, default=None
        A callable that takes in a PIL image and returns a transformed version
        of the image as a PyTorch tensor.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test", "bbox"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        assert split in ["train", "test", "bbox"], f"split {split} is not available."
        assert callable(transform) or transform is None, (
            "transform is not callable or None."
        )

        data_path = os.path.join(root_dir, split + "_data.json")

        assert os.path.isfile(data_path), f"entries file does not exist: {data_path}."

        with open(data_path, "rb") as file:
            entries = json.load(file)
        self.entries = entries

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])

        self.bbox = split == "bbox"

    def __getitem__(self, idx: int) -> Example:
        """Return image-label or image-label-tabular(bbox)."""
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(entry["label"])

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label,
                "qid": entry["qid"],
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if self.bbox:
            example["bbox"] = entry["bbox"]

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)
