"""NCK CRC Dataset."""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from typing import Callable, Optional, Union, Dict
import random

from omegaconf import MISSING
from mmlearn.conf import external_store
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES
from datasets import load_from_disk, load_dataset


@external_store(group="datasets", root_dir=os.getenv("NCK_CRC_ROOT_DIR", MISSING))
class NCK_CRC(Dataset[Example]):
    """NCK CRC dataset for colorectal cancer classification.
    
    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str
        Dataset split, one of 'train', 'train_nonorm', or 'validation'.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "validation",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        processor: Optional[
            Callable[[torch.Tensor, str], tuple[torch.Tensor, str]]
        ] = None,
    ) -> None:
        """Initialize the NCK CRC dataset."""
        assert split in ("train", "train_nonorm", "validation"), f"Invalid split: {split}"
        
        # Class mapping for labels
        self.class_mapping = {
            "ADI": 0,
            "DEB": 1,
            "LYM": 2,
            "MUC": 3,
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
            dataset = dataset.filter(lambda row: row["label"] != "BACK")  # Exclude "BACK" label
            dataset.save_to_disk(cache_path)

        self.data = dataset

        if processor is None and transform is None:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])
        elif processor is None:
            self.transform = transform
        else:
            self.transform = None

        self.tokenizer = tokenizer
        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]
        label = self.class_mapping[entry["label"]]
        label_description = self.get_label_mapping()[label]
        description = random.choice(TEMPLATES[self.__class__.__name__])(label_description)
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        if self.transform is not None:
            image = self.transform(image)

        if self.processor is not None:
            image, tokens = self.processor(image, label)

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: label_description,
                Modalities.RGB.target: label,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert Modalities.TEXT in tokens, f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def get_label_mapping(self):
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

    def name(self) -> str:
        """Return the name of the dataset."""
        return "NCK_CRC"
