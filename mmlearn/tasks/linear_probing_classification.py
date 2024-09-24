"""Zero-shot and linear probing classification evaluation task."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
import torchmetrics
from hydra_zen import store
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import random

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.data_collator import collate_example_list
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core.modalities import Modality
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""

    metric_name: str
    query_modality: str
    top_k: List[int]


@store(group="eval_task", provider="mmlearn")
class LinearProbingClassification(EvaluationHooks):
    """
    Zero-shot classification evaluation task.

    This task evaluates the zero-shot classification performance.

    Parameters
    ----------
    task_specs : List[ClassificationTaskSpec]
        A list of classification task specifications.
    """

    def __init__(
        self,
        task_specs: List[ClassificationTaskSpec],
    ):
        super().__init__()
        self.task_specs = task_specs
        self.metrics: Dict[Any, Metric] = {}


    def on_evaluation_epoch_start(
        self, pl_module: LightningModule, all_dataset_info: Dict[str, Any]
    ) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)

        self.all_dataset_info = all_dataset_info

        for dataset_name in list(all_dataset_info.keys()):
            for spec in self.task_specs:
                assert Modalities.has_modality(spec.query_modality)
                query_modality = Modalities.get_modality(spec.query_modality)
                metric_name = spec.metric_name
                metric = getattr(torchmetrics, metric_name)
                num_classes = all_dataset_info[dataset_name].get_class_count()
                task = "multiclass" if num_classes > 2 else "binary"
                self.metrics.update(
                    {
                        (
                            query_modality,
                            metric_name,
                            dataset_name,
                        ): MetricCollection(
                            {
                                f"{query_modality}_C_{metric_name}@{k}_{dataset_name}": (
                                    metric(
                                        **(
                                            {
                                                "top_k": k,
                                                "task": task,
                                                "num_classes": num_classes,
                                            }
                                            if "top_k"
                                            in inspect.signature(metric).parameters
                                            else {
                                                "task": "multiclass",
                                                "num_classes": num_classes,
                                            }
                                        )
                                    ).to(pl_module.device)
                                )
                                for k in spec.top_k
                            }
                        )
                    }
                )

    def evaluation_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Update classification accuracy metrics."""
        if trainer.sanity_checking:
            return

        outputs: Dict[Union[str, Modality], Any] = pl_module(batch)

        for (
            query_modality,
            _,
            dataset_name,
        ), metric in self.metrics.items():
            output_embeddings = outputs[query_modality.embedding]  # Predictions
            label_index = batch[query_modality.target]  # True labels
            names = batch[
                "dataset_index"
            ]

            # Filter indices where dataset name is part of the metric_name
            matching_indices = [
                i for i, name in enumerate(names) if str(name.item()) in dataset_name
            ]

            if not matching_indices:
                continue

            logits = output_embeddings[matching_indices]
            label_index_filtered = label_index[matching_indices]

            metric.update(logits, label_index_filtered)

    def on_evaluation_epoch_end(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Compute the classification accuracy metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        results = {}
        for (
            _,
            _,
            _,
        ), metric in self.metrics.items():
            results.update(metric.compute())
            metric.reset()
        return results
