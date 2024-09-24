"""HAM10000 Dataset."""

import os
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
from mmlearn.constants import EXAMPLE_INDEX_KEY


@external_store(group="datasets", root_dir=os.getenv("HAM10000_ROOT_DIR", MISSING))
class HAM10000(Dataset[Example]):
    """HAM10000 dataset for zero-shot classification.
    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
        processor: Optional[
            Callable[[torch.Tensor, str], tuple[torch.Tensor, str]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, "HAM10000_metadata.csv"))

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
        entry = self.metadata.iloc[idx]
        image_path = os.path.join(
            self.root_dir, "skin_cancer", entry["image_id"] + ".jpg"
        )
        label_index = list(self.get_label_mapping().keys()).index(entry["dx"])
        label = list(self.get_label_mapping().values())[label_index]
        tokens = self.tokenizer(label) if self.tokenizer is not None else None

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
                Modalities.RGB.target: label_index,
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

    def get_label_mapping(self):
        return {
            "nv": "melanocytic nevus",
            "mel": "melanoma",
            "bkl": "benign keratosis",
            "bcc": "basal cell carcinoma",
            "akiec": "actinic keratosis",
            "vasc": "vascular lesion",
            "df": "dermatofibroma",
        }

    def name(self):
        return "HAM10000"
