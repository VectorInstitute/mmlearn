"""Modules for pretraining, downstream and evaluation tasks."""

from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining
from mmlearn.tasks.linear_probing_classification import LinearProbingClassification
from mmlearn.tasks.zero_shot_classification import ZeroShotClassification
from mmlearn.tasks.zero_shot_retrieval import ZeroShotCrossModalRetrieval


__all__ = [
    "ContrastivePretraining",
    "ZeroShotCrossModalRetrieval",
    "ZeroShotClassification",
    "LinearProbingClassification",
]
