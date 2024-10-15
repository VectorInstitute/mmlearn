"""BACH Dataset."""

import os
import random
from typing import Callable, Dict, Optional, Union

import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from datasets import load_dataset
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("BACH_ROOT_DIR", MISSING))
class BACH(Dataset[Example]):
    """BACH dataset for breast cancer classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str
        Dataset split, one of 'train' or 'test'.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
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
        """Initialize the BACH dataset."""
        os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)

        dataset = load_dataset(
            "1aurent/BACH",
            cache_dir=os.path.join(root_dir, "scratch/"),
            split="train",
        )
        data_dict = dataset.train_test_split(
            test_size=0.25, train_size=0.75, shuffle=True, seed=0
        )
        self.data = data_dict[split]

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

        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.data[idx]
        image = entry["image"]
        label = int(entry["label"])
        label_description = self.get_label_mapping()[label]
        description = random.choice(TEMPLATES[self.__class__.__name__])(
            label_description
        )
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        if self.transform is not None:
            image = self.transform(image)

        if self.processor is not None:
            image, tokens = self.processor(image, label)

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.TEXT.name: label_description,
                Modalities.RGB.target: int(entry["label"]),
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT.name in tokens
                ), f"Missing key `{Modalities.TEXT.name}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT.name] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def get_label_mapping(self):
        """Return the label mapping for the BACH dataset."""
        return {
            0: "breast non-malignant benign tissue",
            1: "breast malignant in-situ carcinoma",
            2: "breast malignant invasive carcinoma",
            3: "breast normal breast tissue",
        }

    def name(self) -> str:
        """Return the name of the dataset."""
        return "BACH"
