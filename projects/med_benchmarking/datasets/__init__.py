import os
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from hydra_zen import builds
from omegaconf import MISSING
from timm.data.transforms import ResizeKeepRatio

from mmlearn.conf import external_store

from .bach import BACH
from .ham10000 import HAM10000
from .lc25000 import LC25000
from .med_mnist_plus import MedMNISTPlus
from .medvqa import MedVQA, MedVQAProcessor
from .mimiciv_cxr import MIMICIVCXR
from .nck import NckCrc
from .pad_ufes_20 import PadUfes20
from .pcam import PCAM
from .pmcoa import PMCOA
from .quilt import Quilt
from .roco import ROCO
from .sicap import SICAP


_MedVQAConf = builds(
    MedVQA,
    split="train",
    encoder={"image_size": 224, "feat_dim": 512, "images_filename": "images_clip.pkl"},
    autoencoder={
        "available": True,
        "image_size": 128,
        "feat_dim": 64,
        "images_filename": "images128x128.pkl",
    },
    num_ans_candidates=MISSING,
)
_PathVQAConf = builds(
    MedVQA,
    root_dir=os.getenv("PATHVQA_ROOT_DIR", MISSING),
    num_ans_candidates=3974,
    autoencoder={"available": False},
    builds_bases=(_MedVQAConf,),
)
_VQARADConf = builds(
    MedVQA,
    root_dir=os.getenv("VQARAD_ROOT_DIR", MISSING),
    num_ans_candidates=458,
    autoencoder={"available": False},
    builds_bases=(_MedVQAConf,),
)
external_store(_MedVQAConf, name="MedVQA", group="datasets")
external_store(_PathVQAConf, name="PathVQA", group="datasets")
external_store(_VQARADConf, name="VQARAD", group="datasets")

external_store(MedVQAProcessor, name="MedVQAProcessor", group="datasets/transforms")


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
    if job_type == "train":
        return transforms.Compose(
            [
                ResizeKeepRatio(512, interpolation="bicubic"),
                transforms.RandomCrop(image_crop_size),
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
            ResizeKeepRatio(image_crop_size, interpolation="bicubic"),
            transforms.CenterCrop(image_crop_size),
            transforms.RGB(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


__all__ = [
    "HAM10000",
    "LC25000",
    "MedMNISTPlus",
    "MedVQA",
    "MedVQAProcessor",
    "MIMICIVCXR",
    "PMCOA",
    "Quilt",
    "ROCO",
    "SICAP",
    "PadUfes20",
    "PCAM",
    "NckCrc",
    "BACH",
    "med_clip_vision_transform",
]
