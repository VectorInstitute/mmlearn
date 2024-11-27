"""Modules for pretraining, downstream and evaluation tasks."""

from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining
from mmlearn.tasks.ijepa_pretraining import IJEPA
from mmlearn.tasks.zero_shot_retrieval import ZeroShotCrossModalRetrieval


__all__ = [
    "ContrastivePretraining",
    "IJEPA",
    "ZeroShotCrossModalRetrieval",
]
