import os
import random
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY, TEMPLATES
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("MEDMNISTPLUS_ROOT_DIR", MISSING))
class MedMNISTPlus(Dataset[Example]):
    """MedMNISTPlus dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function to generate textual embeddings.
    """

    def __init__(
        self,
        root_dir: str,
        name: str = "dermamnist",
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
        assert split in [
            "train",
            "val",
            "test",
        ], f"split {split} is not supported in dataset {name}."
        if name is None:
            raise ValueError("Variable name must be given to dataset")
        self.dataset_name = name
        file_name = name + "_224.npz"
        # Load the dataset
        data = np.load(os.path.join(root_dir, file_name), mmap_mode="r")
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]

        # Set transform, tokenizer, and processor
        if processor is None and transform is None:
            self.transform = Compose([Resize(224), ToTensor()])
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
        image = self.images[idx].astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx].astype(int)
        if len(label) == 1:
            label = int(label[0])
        label_index = label
        label = self.get_label_mapping()[label_index]
        description = random.choice(TEMPLATES[self.name()])(label)
        # Tokenize the label if tokenizer is provided
        tokens = self.tokenizer(description) if self.tokenizer is not None else None

        if self.processor is not None:
            image, tokens = self.processor(image, str(label))

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: str(label),
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
        return self.images.shape[0]

    def name(self):
        """Return the name of the dataset."""
        return self.dataset_name

    def get_label_mapping(self):
        """Return the label mapping based on the dataset name."""
        if self.dataset_name.lower() == "pathmnist":
            return_value = {
                0: "adipose",
                1: "background",
                2: "debris",
                3: "lymphocytes",
                4: "mucus",
                5: "smooth muscle",
                6: "normal colon mucosa",
                7: "cancer-associated stroma",
                8: "colorectal adenocarcinoma epithelium",
            }
        elif self.dataset_name.lower() == "chestmnist":
            return_value = {
                0: "atelectasis",
                1: "cardiomegaly",
                2: "effusion",
                3: "infiltration",
                4: "mass",
                5: "nodule",
                6: "pneumonia",
                7: "pneumothorax",
                8: "consolidation",
                9: "edema",
                10: "emphysema",
                11: "fibrosis",
                12: "pleural",
                13: "hernia",
            }
        elif self.dataset_name.lower() == "dermamnist":
            return_value = {
                0: "actinic keratoses and intraepithelial carcinoma",
                1: "basal cell carcinoma",
                2: "benign keratosis-like lesions",
                3: "dermatofibroma",
                4: "melanoma",
                5: "melanocytic nevi",
                6: "vascular lesions",
            }
        elif self.dataset_name.lower() == "octmnist":
            return_value = {
                0: "choroidal neovascularization",
                1: "diabetic macular edema",
                2: "drusen",
                3: "normal",
            }
        elif self.dataset_name.lower() == "pneumoniamnist":
            return_value = {
                0: "normal",
                1: "pneumonia",
            }
        elif self.dataset_name.lower() == "retinamnist":
            return_value = {
                0: "no apparent retinopathy",
                1: "mild NPDR, non-proliferative diabetic retinopathy",
                2: "moderate NPDR, non-proliferative diabetic retinopathy",
                3: "severe NPDR, non-proliferative diabetic retinopathy",
                4: "PDR, proliferative diabetic retinopathy",
            }
        elif self.dataset_name.lower() == "breastmnist":
            return_value = {
                0: "malignant",
                1: "normal, benign",
            }
        elif self.dataset_name.lower() == "bloodmnist":
            return_value = {
                0: "basophil",
                1: "eosinophil",
                2: "erythroblast",
                3: "immature granulocytes (myelocytes, metamyelocytes, and promyelocytes)",
                4: "lymphocyte",
                5: "monocyte",
                6: "neutrophil",
                7: "platelet",
            }
        elif self.dataset_name.lower() == "tissuemnist":
            return_value = {
                0: "Collecting Duct, Connecting Tubule",
                1: "Distal Convoluted Tubule",
                2: "Glomerular endothelial cells",
                3: "Interstitial endothelial cells",
                4: "Leukocytes",
                5: "Podocytes",
                6: "Proximal Tubule Segments",
                7: "Thick Ascending Limb",
            }
        elif (
            self.dataset_name.lower() == "organsmnist"
            or self.dataset_name.lower() == "organcmnist"
            or self.dataset_name.lower() == "organamnist"
        ):
            return_value = {
                0: "bladder",
                1: "femur-left",
                2: "femur-right",
                3: "heart",
                4: "kidney-left",
                5: "kidney-right",
                6: "liver",
                7: "lung-left",
                8: "lung-right",
                9: "pancreas",
                10: "spleen",
            }
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")
        return return_value
