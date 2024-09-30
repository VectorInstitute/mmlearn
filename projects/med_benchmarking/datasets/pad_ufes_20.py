"""PadUfes20 Dataset."""
import os
import pickle
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from typing import Callable, Optional, Union, Dict
import torch
import random

from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES


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
        split: str,
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
            self.metadata = pickle.load(open(cache_path, "rb"))
        else:
            os.makedirs("cache/", exist_ok=True)
            self.metadata = self._load_and_process_metadata()
            pickle.dump(self.metadata.to_dict("records"), open(cache_path, "wb"))

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
        df["path"] = df["img_id"].apply(lambda imgid: os.path.join(self.root_dir, "Dataset", imgid))
        df.drop(columns=["img_id", "diagnostic"], inplace=True)

        # Split into train and test
        dataset = {}
        dataset["test"] = df.sample(frac=0.2)
        dataset["train"] = df.drop(dataset["test"].index)
        return dataset[self.split]

    def _build_label(self, str_label: str) -> int:
        """Convert diagnostic string label to integer label."""
        classes = {
            "BCC": 0,
            "MEL": 1,
            "SCC": 2,
            "ACK": 3,
            "NEV": 4,
            "SEK": 5
        }
        return classes[str_label]

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.metadata.iloc[idx]
        image_path = entry["path"]
        label = entry["label"]
        description = random.choice(TEMPLATES[self.__class__.__name__])(label)
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        with Image.open(image_path) as img:
            image = img.convert("RGB")

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
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in tokens
                ), f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata)
    
    def get_label_mapping() -> Dict[str, str]:
        """Return the label mapping for the PadUfes20 dataset."""
        return {
            "BCC": "Basal Cell Carcinoma",
            "MEL": "Melanoma",
            "SCC": "Squamous Cell Carcinoma",
            "ACK": "Actinic Keratosis",
            "NEV": "Nevus",
            "SEK": "Seborrheic Keratosis",
        }
