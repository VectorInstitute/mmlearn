import os
from typing import Literal

from hydra_zen import builds, store
from omegaconf import MISSING
from timm.data.transforms import ResizeKeepRatio
from torchvision import transforms

from mmlearn.conf import external_store
from projects.med_benchmarking.datasets.medvqa import MedVQA, MedVQAProcessor
from projects.med_benchmarking.datasets.mimiciv_cxr import MIMICIVCXR
from projects.med_benchmarking.datasets.pmcoa import PMCOA
from projects.med_benchmarking.datasets.quilt import Quilt
from projects.med_benchmarking.datasets.roco import ROCO


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
