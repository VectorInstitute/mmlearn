"""CheXpert Dataset."""

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


_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


@store(
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("CHEXPERT_ROOT_DIR", MISSING),
    split="train",
)
class CheXpert(Dataset[Example]):
    """CheXpert dataset.

    Each datapoint is a pair of `(image, target label)`.

    Parameters
    ----------
    root_dir : str
        Directory which contains `.json` files stating all dataset entries.
    split : {"train", "valid"}
        Dataset split.
    labeler : Optional[{"chexpert", "chexbert", "vchexbert"}], optional, default=None
        Labeler used to extract labels from the training images. "valid" split
        has no labeler, labeling for valid split was done by human radiologists.
    transform : Optional[Callable[[PIL.Image], torch.Tensor], optional, default=None
        A callable that takes in a PIL image and returns a transformed version
        of the image as a PyTorch tensor.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid"],
        labeler: Optional[Literal["chexpert", "chexbert", "vchexbert"]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        assert split in ["train", "valid"], f"split {split} is not available."
        assert labeler in ["chexpert", "chexbert", "vchexbert"] or labeler is None, (
            f"labeler {labeler} is not available."
        )
        assert callable(transform) or transform is None, (
            "transform is not callable or None."
        )

        if split == "valid":
            data_file = f"{split}_data.json"
        elif split == "train":
            data_file = f"{labeler}_{split}_data.json"
        data_path = os.path.join(root_dir, data_file)

        assert os.path.isfile(data_path), f"entries file does not exist: {data_path}."

        with open(data_path, "rb") as file:
            entries = json.load(file)
        self.entries = entries

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th datapoint."""
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(entry["label"])

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label,
                "qid": entry["qid"],
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)
