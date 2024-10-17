"""PadUfes20 Dataset."""

import os
import pickle
import random
from typing import Callable, Dict, Optional, Union

import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from mmlearn.conf import external_store


@external_store(group="datasets", root_dir=os.getenv("PADUFES_ROOT_DIR", MISSING))
class PadUfes20(Dataset[Example]):
    """PadUfes20 dataset for classification tasks.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    split : str
        Dataset split, must be one of ["train", "test"].
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    processor : Optional[Callable], default=None
        Optional processor for post-processing.
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
        """Initialize the dataset."""
        assert split in ["train", "test"], f"split {split} is not supported in dataset."

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor

        # Load cached data if available
        cache_path = f"cache/PadUfes20_{split}.pkl"
        if os.path.exists(cache_path):
            print(f"!!! Using cached dataset for {split}")
            with open(cache_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            os.makedirs("cache/", exist_ok=True)
            self.metadata = self._load_and_process_metadata()
            with open(cache_path, "wb") as f:
                pickle.dump(self.metadata.to_dict("records"), f)

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

    def _load_and_process_metadata(self) -> pd.DataFrame:
        """Load and process metadata from CSV."""
        df = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        df = df[["img_id", "diagnostic"]]
        df["label"] = df["diagnostic"].apply(self._build_label)
        df["path"] = df["img_id"].apply(
            lambda imgid: os.path.join(self.root_dir, "Dataset", imgid)
        )
        df.drop(columns=["img_id", "diagnostic"], inplace=True)

        # Split into train and test
        dataset = {}
        dataset["test"] = df.sample(frac=0.2)
        dataset["train"] = df.drop(dataset["test"].index)
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
    def label_mapping(self) -> Dict[str, str]:
        """Return the label mapping for the PadUfes20 dataset."""
        return {
            "BCC": "Basal Cell Carcinoma",
            "MEL": "Melanoma",
            "SCC": "Squamous Cell Carcinoma",
            "ACK": "Actinic Keratosis",
            "NEV": "Nevus",
            "SEK": "Seborrheic Keratosis",
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
