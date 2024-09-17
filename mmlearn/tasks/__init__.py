"""Modules for pretraining, downstream and evaluation tasks."""

from mmlearn.tasks.classification import Classification
from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining
from mmlearn.tasks.zero_shot_retrieval import ZeroShotCrossModalRetrieval


__all__ = [
    "ContrastivePretraining",
    "ZeroShotCrossModalRetrieval",
    "Classification",
]
