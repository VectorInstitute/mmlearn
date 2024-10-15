from typing import Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.datasets.core import Modalities


class ToyDatasetZSC(Dataset):
    """
    A toy LC25000 dataset for zero-shot classification testing.
    Generates synthetic data rather than reading from disk.
    """

    def __init__(self, size=10, organ="colon", transform=None):
        """
        Initialize the dataset with a given size and organ type.

        Parameters
        ----------
        - size (int): Number of examples in the dataset.
        - organ (str): 'lung' or 'colon', defaults to 'colon'.
        - transform (callable): Transformations to apply to the images.
        """
        self.organ = organ
        self.size = size
        self.data = []
        self.label_mapping = {
            "lung": {
                0: "lung adenocarcinoma",
                1: "lung squamous cell carcinoma",
                2: "lung benign tissue",
            },
            "colon": {0: "colon adenocarcinoma", 1: "colon benign tissue"},
        }[self.organ]

        self.transform = (
            transform
            if transform
            else Compose([Resize(224), CenterCrop(224), ToTensor()])
        )

        # Generate synthetic images (224x224 RGB) and labels
        for _ in range(size):
            image = Image.fromarray(
                np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            )
            label = np.random.choice(list(self.label_mapping.keys()))
            self.data.append({"image": image, "label": label})

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = entry["image"]
        label = entry["label"]

        if self.transform:
            image = self.transform(image)

        return {
            Modalities.RGB: image,
            Modalities.RGB.target: label,
            "dataset_index": idx,  # Simulate EXAMPLE_INDEX_KEY functionality
        }

    @property
    def zero_shot_prompt_templates(self):
        return ["This is an example of {}.", "This is another example of {}."]

    @property
    def label_mapping(self) -> Dict[int, str]:
        return {0: "Class 1", 1: "Class 2", 2: "Class 3"}
