"""SUN RGB-D dataset."""

import os
from typing import Callable, List, Literal, Optional

import numpy as np
import torch
from hydra_zen import MISSING, store
from lightning_utilities.core.imports import RequirementCache
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import to_pil_image

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


_OPENCV_AVAILABLE = RequirementCache("opencv-python>=4.10.0.84")
if _OPENCV_AVAILABLE:
    import cv2  # noqa: F401

_LABELS = [
    "bathroom",
    "bedroom",
    "classroom",
    "computer room",
    "conference room",
    "corridor",
    "dining area",
    "dining room",
    "discussion area",
    "furniture store",
    "home office",
    "kitchen",
    "lab",
    "lecture theatre",
    "library",
    "living room",
    "office",
    "rest space",
    "study space",
]


def text_labels() -> List[str]:
    """Return a list of labels."""
    return _LABELS


# from https://github.com/facebookresearch/omnivore/issues/12#issuecomment-1070911016
sensor_to_params = {
    "kv1": {
        "baseline": 0.075,
    },
    "kv1_b": {
        "baseline": 0.075,
    },
    "kv2": {
        "baseline": 0.075,
    },
    "realsense": {
        "baseline": 0.095,
    },
    "xtion": {
        "baseline": 0.095,  # guessed based on length of 18cm for ASUS xtion v1
    },
}


def convert_depth_to_disparity(
    depth_file: str,
    intrinsics_file: str,
    sensor_type: str,
    min_depth: float = 0.01,
    max_depth: int = 50,
) -> torch.Tensor:
    """Load depth file and convert to disparity.

    Parameters
    ----------
    depth_file : str
        Path to the depth file.
    intrinsics_file : str
        Intrinsics_file is a txt file supplied in SUNRGBD with sensor information
        Can be found at the path: os.path.join(root_dir, room_name, "intrinsics.txt")
    sensor_type : str
        Sensor type of the depth file.
    min_depth : float, default=0.01
        Minimum depth value to clip the depth image.
    max_depth : int, default=50
        Maximum depth value to clip the depth image.

    Returns
    -------
    torch.Tensor
        Disparity image from the depth image following the ImageBind implementation.
    """
    with open(intrinsics_file, "r") as fh:
        lines = fh.readlines()
        focal_length = float(lines[0].strip().split()[0])
    baseline = sensor_to_params[sensor_type]["baseline"]
    depth_image = np.array(PILImage.open(depth_file))
    depth = np.array(depth_image).astype(np.float32)
    depth_in_meters = depth / 1000.0
    if min_depth is not None:
        depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
    disparity = baseline * focal_length / depth_in_meters
    return torch.from_numpy(disparity).float()


@store(
    name="SUNRGBD",
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("SUNRGBD_ROOT_DIR", MISSING),
)
class SUNRGBDDataset(Dataset[Example]):
    """SUN RGB-D dataset.

    Repo followed to extract the dataset:
    https://github.com/TUI-NICR/nicr-scene-analysis-datasets

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset.
    split : {"train", "test"}, default="train"
        Split of the dataset to use.
    return_type : {"disparity", "image"}, default="disparity"
        Return type of the depth images. If "disparity", the depth images are
        converted to disparity similar to the ImageBind implementation.
        Else returns the depth image as a 3-channel image.
    rgb_transform : Optional[Callable[[PILImage], torch.Tensor]], default=None
        Transformation to apply to RGB images.
    depth_transform : Optional[Callable[[PILImage], torch.Tensor]], default=None
        Transformation to apply to depth images.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"] = "train",
        return_type: Literal["disparity", "image"] = "disparity",
        rgb_transform: Optional[Callable[[PILImage], torch.Tensor]] = None,
        depth_transform: Optional[Callable[[PILImage], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if not _OPENCV_AVAILABLE:
            raise ImportError(
                "SUN RGB-D dataset requires `opencv-python` which is not installed.",
            )

        self._validate_args(root_dir, split, rgb_transform, depth_transform)
        self.return_type = return_type

        self.root_dir = root_dir
        with open(os.path.join(root_dir, f"{split}.txt"), "r") as f:
            file_ids = f.readlines()
        file_ids = [f.strip() for f in file_ids]

        root_dir = os.path.join(root_dir, split)
        depth_files = [os.path.join(root_dir, "depth", f"{f}.png") for f in file_ids]
        rgb_files = [os.path.join(root_dir, "rgb", f"{f}.jpg") for f in file_ids]
        intrinsic_files = [
            os.path.join(root_dir, "intrinsics", f"{f}.txt") for f in file_ids
        ]

        sensor_types = [
            file.removeprefix(os.path.join(root_dir, "depth")).split(os.sep)[1]
            for file in depth_files
        ]

        label_files = [
            os.path.join(root_dir, "scene_class", f"{f}.txt") for f in file_ids
        ]
        labels = []
        for label_file in label_files:
            with open(label_file, "r") as file:  # noqa: SIM115
                labels.append(file.read().strip())
        labels = [label.replace("_", " ") for label in labels]
        labels = [
            _LABELS.index(label) if label in _LABELS else len(_LABELS)  # type: ignore
            for label in labels
        ]

        # remove the samples with classes not in _LABELS
        # this is to follow the same classes used in ImageBind
        if split == "test":
            valid_indices = [
                i
                for i, label in enumerate(labels)
                if label < len(_LABELS)  # type: ignore
            ]
            rgb_files = [rgb_files[i] for i in valid_indices]
            depth_files = [depth_files[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            intrinsic_files = [intrinsic_files[i] for i in valid_indices]
            sensor_types = [sensor_types[i] for i in valid_indices]

        self.samples = list(
            zip(rgb_files, depth_files, labels, intrinsic_files, sensor_types)
        )

        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.samples)

    def _validate_args(
        self,
        root_dir: str,
        split: str,
        rgb_transform: Optional[Callable[[PILImage], torch.Tensor]],
        depth_transform: Optional[Callable[[PILImage], torch.Tensor]],
    ) -> None:
        """Validate arguments."""
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(
                f"The given `root_dir` {root_dir} is not a directory",
            )
        if split not in ["train", "test"]:
            raise ValueError(
                f"Expected `split` to be one of `'train'` or `'test'`, but got {split}",
            )
        if rgb_transform is not None and not callable(rgb_transform):
            raise TypeError(
                f"Expected argument `rgb_transform` to be callable, but got {type(rgb_transform)}",
            )
        if depth_transform is not None and not callable(depth_transform):
            raise TypeError(
                f"Expected `depth_transform` to be callable, but got {type(depth_transform)}",
            )

    def __getitem__(self, idx: int) -> Example:
        """Return RGB and depth images at index `idx`."""
        # Read images
        rgb_image = cv2.imread(self.samples[idx][0], cv2.IMREAD_UNCHANGED)
        if self.rgb_transform is not None:
            rgb_image = self.rgb_transform(to_pil_image(rgb_image))

        if self.return_type == "disparity":
            depth_image = convert_depth_to_disparity(
                self.samples[idx][1],
                self.samples[idx][3],
                self.samples[idx][4],
            )
        else:
            # Using cv2 instead of PIL Image since we use PNG grayscale images.
            depth_image = cv2.imread(
                self.samples[idx][1],
                cv2.IMREAD_GRAYSCALE,
            )
            # Make a 3-channel depth image to enable passing to a pretrained ViT.
            depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=-1)

        if self.depth_transform is not None:
            depth_image = self.depth_transform(to_pil_image(depth_image))

        return Example(
            {
                Modalities.RGB.name: rgb_image,
                Modalities.DEPTH.name: depth_image,
                EXAMPLE_INDEX_KEY: idx,
                Modalities.DEPTH.target: self.samples[idx][2],
            }
        )
