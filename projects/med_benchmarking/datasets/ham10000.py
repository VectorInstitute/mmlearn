import os
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("HAM10000_ROOT_DIR", MISSING))
class HAM10000(Dataset[Example]):
    """HAM10000 dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the HAM10000 dataset."""
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, "HAM10000_metadata.csv"))

        self.transform = (
            Compose([Resize(224), CenterCrop(224), ToTensor()])
            if transform is None
            else transform
        )

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image",
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.metadata.iloc[idx]
        image_path = os.path.join(
            self.root_dir, "skin_cancer", f"{entry['image_id']}.jpg"
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_index = list(self.label_mapping.keys()).index(entry["dx"])

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label_index,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    @property
    def label_mapping(self) -> Dict[str, str]:
        """Return the label mapping."""
        return {
            "nv": "Melanocytic Nevi",
            "mel": "Melanoma",
            "bkl": "Benign Keratosis-like Lesions",
            "bcc": "Basal Cell Carcinoma",
            "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
            "vasc": "Vascular Lesions",
            "df": "Dermatofibroma",
        }
