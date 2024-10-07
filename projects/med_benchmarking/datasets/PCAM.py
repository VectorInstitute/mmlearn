"""PCAM Dataset."""

import os
import pickle
import random
from typing import Callable, Optional, Union, Dict
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from omegaconf import MISSING
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES
from mmlearn.conf import external_store

@external_store(group="datasets", root_dir=os.getenv("PCAM_ROOT_DIR", MISSING))
class PCAM(Dataset[Example]):
    """PCAM dataset for classification tasks.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory or cache directory.
    split : str
        Dataset split (e.g., 'train', 'test', 'validation').
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    processor : Optional[Callable], default=None
        Function to process image and token pairs.
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
        """Initialize the PCAM dataset."""
        cache_path = os.path.join(root_dir, f"cache/pcam_{split}.pkl")

        if os.path.exists(cache_path):
            print("!!!Using cached dataset")
            self.data = pickle.load(open(cache_path, "rb"))
        else:
            os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)
            dataset = load_dataset(
                "1aurent/PatchCamelyon",
                cache_dir=os.path.join(root_dir, "scratch/")
            )[split]

            self.data = dataset
            pickle.dump(self.data, open(cache_path, "wb"))

        self.transform = transform or Compose([Resize(224), CenterCrop(224), ToTensor()])
        self.tokenizer = tokenizer
        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        print()
        entry = self.data[idx]
        image = entry["image"].convert("RGB")
        label_idx = int(entry["label"])
        label = list(self.get_label_mapping().values())[label_idx]
        print(f"label_idx :::::::: {label_idx}")
        print(f"label :::::::: {label}")

        if self.transform is not None:
            image = self.transform(image)
        
        if self.tokenizer is not None:
            tokens = self.tokenizer(str(label))
            description = random.choice(TEMPLATES[self.__class__.__name__])(label)
            tokens = self.tokenizer(description) if self.tokenizer is not None else None

        if self.processor is not None:
            image, label = self.processor(image, str(label))

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: str(label),
                Modalities.RGB.target: label,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if isinstance(tokens, dict):
            example.update(tokens)
        else:
            example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
    
    def get_label_mapping(self) -> str:
        """Return the mapping of labels for the PCAM dataset."""
        return {
            0: "lymph node",
            1: "lymph node containing metastatic tumor tissue"
        }
