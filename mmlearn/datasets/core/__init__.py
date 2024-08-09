"""Modules for core dataloading functionality."""

from mmlearn.datasets.core.combined_dataset import CombinedDataset
from mmlearn.datasets.core.example import (
    Example,
    collate_example_list,
    find_matching_indices,
)
from mmlearn.datasets.core.modalities import ModalityRegistry
from mmlearn.datasets.core.samplers import (
    CombinedDatasetRatioSampler,
    DistributedEvalSampler,
)


Modalities = ModalityRegistry()

__all__ = [
    "CombinedDataset",
    "Example",
    "collate_example_list",
    "find_matching_indices",
    "CombinedDatasetRatioSampler",
    "DistributedEvalSampler",
    "Modalities",
]
