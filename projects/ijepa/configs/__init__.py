import os
from typing import Literal

from hydra_zen import builds
from omegaconf import MISSING
from timm.data.transforms import ResizeKeepRatio
from torchvision import transforms

from mmlearn.conf import external_store

@external_store(group="datasets/transforms")
def med_clip_vision_transform(
    image_crop_size: int = 224, job_type: Literal["train", "eval"] = "train"
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    return transforms.Compose(
        [
            ResizeKeepRatio(
                512 if job_type == "train" else image_crop_size, interpolation="bicubic"
            ),
            transforms.RandomCrop(image_crop_size)
            if job_type == "train"
            else transforms.CenterCrop(image_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
