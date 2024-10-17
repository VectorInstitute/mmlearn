import os
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("MEDMNISTPLUS_ROOT_DIR", MISSING))
class MedMNISTPlus(Dataset[Example]):
    """MedMNISTPlus dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata.
    name : str
        Specific name of the MedMNIST dataset variant.
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        name: str = "organamnist",
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

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}.",
            "histopathology image of {}.",
            "pathology tissue showing {}.",
            "presence of {} tissue on image.",
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        image = self.images[idx].astype(np.uint8)
        image = self.transform(Image.fromarray(image).convert("RGB"))
        label = self.labels[idx].astype(int)
        if len(label) == 1:
            label = int(label[0])

        if self.transform is not None:
            image = self.transform(image)

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    @property
    def label_mapping(self) -> Dict[int, str]:
        """Return the label mapping based on the dataset name."""
        """
        if self.name == "pathmnist":
            return {
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
        elif self.name == "chestmnist":
            return {
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
        elif self.name == "dermamnist":
            return {
                0: "actinic keratoses and intraepithelial carcinoma",
                1: "basal cell carcinoma",
                2: "benign keratosis-like lesions",
                3: "dermatofibroma",
                4: "melanoma",
                5: "melanocytic nevi",
                6: "vascular lesions",
            }
        elif self.name == "octmnist":
            return {
                0: "choroidal neovascularization",
                1: "diabetic macular edema",
                2: "drusen",
                3: "normal",
            }
        elif self.name == "pneumoniamnist":
            return {
                0: "normal",
                1: "pneumonia",
            }
        elif self.name == "retinamnist":
            return {
                0: "no apparent retinopathy",
                1: "mild NPDR, non-proliferative diabetic retinopathy",
                2: "moderate NPDR, non-proliferative diabetic retinopathy",
                3: "severe NPDR, non-proliferative diabetic retinopathy",
                4: "PDR, proliferative diabetic retinopathy",
            }
        elif self.name == "breastmnist":
            return {
                0: "malignant",
                1: "normal, benign",
            }
        elif self.name == "bloodmnist":
            return {
                0: "basophil",
                1: "eosinophil",
                2: "erythroblast",
                3: "immature granulocytes (myelocytes, metamyelocytes, " \
                    "and promyelocytes)",
                4: "lymphocyte",
                5: "monocyte",
                6: "neutrophil",
                7: "platelet",
            }
        elif self.name == "tissuemnist":
            return {
                0: "Collecting Duct, Connecting Tubule",
                1: "Distal Convoluted Tubule",
                2: "Glomerular endothelial cells",
                3: "Interstitial endothelial cells",
                4: "Leukocytes",
                5: "Podocytes",
                6: "Proximal Tubule Segments",
                7: "Thick Ascending Limb",
            }
        elif self.name == "organsmnist":
            return {
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
        """
        return {
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
