import os
from typing import Literal
from logging import getLogger

from PIL import ImageFilter

import torch
from torchvision import transforms
from mmlearn.conf import external_store
from timm.data.transforms import ResizeKeepRatio


logger = getLogger()


@external_store(group="datasets/transforms")
def linear_eval_transforms(
    crop_size: int = 224,
    normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    job_type: Literal["train", "eval"] = "train",
) -> transforms.Compose:
    """
    Create transforms for linear evaluation.

    Parameters
    ----------
    crop_size : int, default=224
        Size of the image crop.
    normalization : tuple, default=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Mean and std for normalization.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for linear evaluation with images.
    """
    transforms_list = []
    if job_type == "train":
        transforms_list.append(transforms.RandomResizedCrop(crop_size))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(ResizeKeepRatio(crop_size + 32, interpolation="bicubic"))
        transforms_list.append(transforms.CenterCrop(crop_size))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(normalization[0], normalization[1]))

    return transforms.Compose(transforms_list)


@external_store(group="datasets/transforms")
def ijepa_transforms(
    crop_size: int = 224,
    crop_scale: tuple = (0.3, 1.0),
    color_jitter_strength: float = 0.0,
    horizontal_flip: bool = False,
    color_distortion: bool = False,
    gaussian_blur: bool = False,
    normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    job_type: Literal["train", "eval"] = "train",
) -> transforms.Compose:
    """
    Create transforms for training and evaluation.

    Parameters
    ----------
    crop_size : int, default=224
        Size of the image crop.
    crop_scale : tuple, default=(0.3, 1.0)
        Range for the random resized crop scaling.
    color_jitter_strength : float, default=0.0
        Strength of color jitter.
    horizontal_flip : bool, default=False
        Whether to apply random horizontal flip.
    color_distortion : bool, default=False
        Whether to apply color distortion.
    gaussian_blur : bool, default=False
        Whether to apply Gaussian blur.
    normalization : tuple, default=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Mean and std for normalization.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training/evaluation with images.
    """
    logger.info("Creating data transforms")

    def get_color_distortion(s: float = 1.0):
        """Apply color jitter and random grayscale."""
        color_jitter_transform = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        return transforms.Compose(
            [
                transforms.RandomApply([color_jitter_transform], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    class GaussianBlur:
        """Gaussian blur transform."""

        def __init__(
            self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
        ):
            self.prob = p
            self.radius_min = radius_min
            self.radius_max = radius_max

        def __call__(self, img):
            if torch.bernoulli(torch.tensor(self.prob)) == 0:
                return img
            radius = self.radius_min + torch.rand(1).item() * (
                self.radius_max - self.radius_min
            )
            return img.filter(ImageFilter.GaussianBlur(radius))

    transforms_list = []
    if job_type == "train":
        transforms_list.append(
            transforms.RandomResizedCrop(crop_size, scale=crop_scale)
        )
        if horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if color_distortion:
            transforms_list.append(get_color_distortion(s=color_jitter_strength))
        if gaussian_blur:
            transforms_list.append(GaussianBlur(p=0.5))
    else:
        transforms_list.append(transforms.Resize(crop_size))
        transforms_list.append(transforms.CenterCrop(crop_size))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(normalization[0], normalization[1]))

    return transforms.Compose(transforms_list)
