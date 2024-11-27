"""PadUfes20 Dataset."""

import os
import pickle
from typing import Callable, Dict, Literal, Optional

import pandas as pd
import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


@external_store(group="datasets", root_dir=os.getenv("PADUFES_ROOT_DIR", MISSING))
class PadUfes20(Dataset[Example]):
    """PadUfes20 dataset for classification tasks.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    split : {'train', 'test'}
        Dataset split, must be one of ["train", "test"].
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the dataset."""
        assert split in ["train", "test"], f"split {split} is not supported in dataset."

        self.root_dir = root_dir
        self.split = split

        # Load cached data if available
        cache_path = f".cache/PadUfes20_{split}.pkl"
        if os.path.exists(cache_path):
            print(f"!!! Using cached dataset for {split}")
            with open(cache_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            os.makedirs(".cache/", exist_ok=True)
            self.metadata = self._load_and_process_metadata()
            with open(cache_path, "wb") as f:
                pickle.dump(self.metadata.to_dict("records"), f)

        self.transform = (
            Compose([Resize(224), CenterCrop(224), ToTensor()])
            if transform is None
            else transform
        )

    def _load_and_process_metadata(self) -> pd.DataFrame:
        """Load and process metadata from CSV."""
        df = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        df = df[["img_id", "diagnostic"]]
        df["label"] = df["diagnostic"].apply(self._build_label)
        df["path"] = df["img_id"].apply(
            lambda imgid: os.path.join(self.root_dir, "Dataset", imgid)
        )
        df.drop(columns=["img_id", "diagnostic"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Split into train and test
        dataset = {}
        dataset["test"] = df.sample(frac=0.2, ignore_index=True)
        dataset["train"] = df.drop(dataset["test"].index).reset_index(drop=True)
        return dataset[self.split]

    def _build_label(self, str_label: str) -> int:
        """Convert diagnostic string label to integer label."""
        classes = {"BCC": 0, "MEL": 1, "SCC": 2, "ACK": 3, "NEV": 4, "SEK": 5}
        return classes[str_label]

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.metadata[idx]
        image_path = entry["path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

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
        return len(self.metadata)

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the label mapping for the PadUfes20 dataset."""
        return {
            0: "Basal Cell Carcinoma",
            1: "Melanoma",
            2: "Squamous Cell Carcinoma",
            3: "Actinic Keratosis",
            4: "Nevus",
            5: "Seborrheic Keratosis",
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
