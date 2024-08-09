"""ImageNet dataset."""

import os
from typing import Any, Callable, Literal, Optional

from hydra_zen import MISSING, store
from torchvision.datasets.folder import ImageFolder

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@store(
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("IMAGENET_ROOT_DIR", MISSING),
)
class ImageNet(ImageFolder):
    """ImageNet dataset.

    This is a wrapper around `torchvision.datasets.ImageFolder` that returns a
    `dataset.example.Example` object.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset.
    split : {"train", "val"}, default="train"
        The split of the dataset to use.
    transform : Callable, optional, default=None
        A function/transform that takes in an PIL image and returns a transformed
        version.
    target_transform : Callable, optional, default=None
        A function/transform that takes in the target and transforms it.
    mask_generator : Optional[Callable[..., Any]], optional, default=None
        Generator for the mask.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable[..., Any]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        mask_generator: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the dataset."""
        split = "train" if split == "train" else "val"
        root_dir = os.path.join(root_dir, split)
        super().__init__(
            root=root_dir, transform=transform, target_transform=target_transform
        )
        self.mask_generator = mask_generator

    def __getitem__(self, index: int) -> Example:
        """Get an example at the given index."""
        image, target = super().__getitem__(index)
        example = Example(
            {
                Modalities.RGB: image,
                Modalities.RGB.target: target,
                EXAMPLE_INDEX_KEY: index,
            }
        )
        mask = self.mask_generator() if self.mask_generator else None
        if mask is not None:  # error will be raised during collation if `None`
            example[Modalities.RGB.mask] = mask
        return example
