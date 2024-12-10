from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from timm.data.transforms import RandomResizedCropAndInterpolation, ResizeKeepRatio

from mmlearn.conf import external_store


@external_store(group="datasets/transforms")
def rgb_transform(
    resize_to: int = 256,
    crop_size: int = 224,
    job_type: Literal["train", "eval"] = "train",
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Parameters
    ----------
    resize_to : int, default=256
        Size to which the image should be resized prior to cropping.
    crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    if job_type == "train":
        return transforms.Compose(
            [
                RandomResizedCropAndInterpolation(crop_size, interpolation="bicubic"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomErasing(p=0.25),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RGB(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    return transforms.Compose(
        [
            ResizeKeepRatio(resize_to, interpolation="bicubic"),
            transforms.CenterCrop(crop_size),
            transforms.RGB(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


@external_store(group="datasets/transforms")
def depth_transform(
    resize_to: int = 256,
    crop_size: int = 224,
    norm_mean: float = 0.02,  # for nyu_v2 dataset
    norm_std: float = 0.00295,  # for nyu_v2 dataset
    job_type: Literal["train", "eval"] = "train",
):
    norm_op = transforms.Normalize(mean=(norm_mean,), std=(norm_std,))
    if job_type == "train":
        return transforms.Compose(
            [
                RandomResizedCropAndInterpolation(crop_size, interpolation="bicubic"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                norm_op,
            ]
        )

    return transforms.Compose(
        [
            ResizeKeepRatio(resize_to, interpolation="bicubic"),
            transforms.CenterCrop(crop_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            norm_op,
        ]
    )
