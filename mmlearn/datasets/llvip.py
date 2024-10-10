"""LLVIP dataset."""

import glob
import os
import xml.etree.ElementTree as ET  # noqa: N817
from typing import Callable, Dict, Optional

import numpy as np
import torch
from hydra_zen import MISSING, store
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@store(
    name="LLVIP",
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("LLVIP_ROOT_DIR", MISSING),
)
class LLVIPDataset(Dataset[Example]):
    """A dataset class for the LLVIP dataset which handles RGB and IR images.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset. The directory should contain
        'visible' and 'infrared' subdirectories.
    train : bool, default=True
        Flag to indicate training or testing phase.
    transform : Optional[Callable], optional, default=None
        Transformations to be applied to the images.
    """

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[Callable[[PILImage], torch.Tensor]] = None,
    ):
        """Initialize the dataset."""
        self.path_images_rgb = os.path.join(
            root_dir,
            "visible",
            "train" if train else "test",
        )
        self.path_images_ir = os.path.join(
            root_dir, "infrared", "train" if train else "test"
        )
        self.train = train
        self.transform = transform or transforms.ToTensor()

        self.rgb_images = sorted(glob.glob(os.path.join(self.path_images_rgb, "*.jpg")))
        self.ir_images = sorted(glob.glob(os.path.join(self.path_images_ir, "*.jpg")))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.rgb_images)

    def __getitem__(self, idx: int) -> Example:
        """Return an example from the dataset."""
        rgb_image_path = self.rgb_images[idx]
        ir_image_path = self.ir_images[idx]

        rgb_image = PILImage.open(rgb_image_path).convert("RGB")
        ir_image = PILImage.open(ir_image_path).convert("L")

        example = Example(
            {
                Modalities.RGB.name: self.transform(rgb_image),
                Modalities.THERMAL.name: self.transform(ir_image),
                EXAMPLE_INDEX_KEY: idx,
            },
        )

        if self.train:
            annot_path = (
                rgb_image_path.replace("visible", "Annotations")
                .replace(".jpg", ".xml")
                .replace("train", "")
            )
            annot = self._get_bbox(annot_path)
            example["annotation"] = {
                "bboxes": torch.from_numpy(annot["bboxes"]),
                "labels": torch.from_numpy(annot["labels"]),
            }
        return example

    def _get_bbox(self, filename: str) -> Dict[str, np.ndarray]:
        """Parse the XML file to get bounding boxes and labels.

        Parameters
        ----------
        filename : str
            Path to the annotation XML file.

        Returns
        -------
        dict
            A dictionary containing bounding boxes and labels.
        """
        try:
            root = ET.parse(filename).getroot()

            bboxes, labels = [], []
            for obj in root.findall("object"):
                bbox_obj = obj.find("bndbox")
                bbox = [
                    int(bbox_obj.find(dim).text)  # type: ignore[union-attr,arg-type]
                    for dim in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bboxes.append(bbox)
                labels.append(1)  # Assuming 'person' is the only label
            return {
                "bboxes": np.array(bboxes).astype("float"),
                "labels": np.array(labels).astype("int"),
            }
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML: {e}") from None
        except Exception as e:
            raise RuntimeError(
                f"Error processing annotation file {filename}: {e}",
            ) from None
