"""Modules for pretraining, downstream and evaluation tasks."""

from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining
from mmlearn.tasks.ijepa import IJEPA
from mmlearn.tasks.zero_shot_classification import ZeroShotClassification
from mmlearn.tasks.zero_shot_retrieval import ZeroShotCrossModalRetrieval
from mmlearn.tasks.linear_probing import LinearClassifierModule


__all__ = [
    "ContrastivePretraining",
    "IJEPA",
    "ZeroShotCrossModalRetrieval",
    "ZeroShotClassification",
    "LinearClassifierModule",
]
