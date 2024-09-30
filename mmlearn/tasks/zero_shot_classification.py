"""Zero-shot classification evaluation task."""

import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from hydra_zen import store
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)

from mmlearn.constants import TEMPLATES
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""

    query_modality: str
    top_k: List[int]


@store(group="eval_task", provider="mmlearn")
class ZeroShotClassification(EvaluationHooks):
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
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ):
        super().__init__()
        self.task_specs = task_specs
        for spec in self.task_specs:
            assert Modalities.has_modality(spec.query_modality)

        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be set in the dataset to generate tokenized label descriptions"
            )
        self.tokenizer = tokenizer

    def set_all_dataset_info(self, all_dataset_info: Dict[str, Any]) -> None:
        """
        Set the dataset information for the task.

        This method assigns the provided dataset information to the `all_dataset_info`
        attribute and initializes the necessary metrics for evaluation.

        Args:
            all_dataset_info (Dict[str, Any], optional): A dictionary containing
            information about all the datasets. Defaults to None.
        """
        self.all_dataset_info = all_dataset_info
        self._create_metrics()

    def _create_metrics(self) -> None:
        # creating metrics
        self.metrics: Dict[Any, Metric] = {}
        all_metrics = {
            "accuracy": Accuracy,
            "f1_score": F1Score,
            "aucroc": AUROC,
            "precision": Precision,
            "recall": Recall,
        }

        for dataset_index in list(self.all_dataset_info.keys()):
            for spec in self.task_specs:
                query_modality = Modalities.get_modality(spec.query_modality)
                num_classes = self.all_dataset_info[dataset_index].get_class_count()
                name = self.all_dataset_info[dataset_index].get_name()
                for metric_name, metric in all_metrics.items():
                    self.metrics.update(
                        {
                            (
                                query_modality,
                                metric_name,
                                dataset_index,
                            ): MetricCollection(
                                {
                                    f"{spec.query_modality}_C_{metric_name}@{k}_{name}": (
                                        metric(
                                            **(
                                                {
                                                    "top_k": k,
                                                    "task": "multiclass",
                                                    "num_classes": num_classes,
                                                }
                                                if "top_k"
                                                in inspect.signature(metric).parameters
                                                else {
                                                    "task": "multiclass",
                                                    "num_classes": num_classes,
                                                }
                                            )
                                        )
                                    )
                                    for k in spec.top_k
                                }
                            )
                        }
                    )

    def on_evaluation_epoch_start(self, pl_module: LightningModule) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)

        # Set the label embeddings - It is better if we can do this at initialization
        templates: Dict[str, Tuple[Callable[[Any], str], ...]] = TEMPLATES
        for dataset_index, dataset_info in self.all_dataset_info.items():
            dataset_name = self.all_dataset_info[dataset_index].get_name()
            selected_template = random.choice(
                templates[dataset_name]
            )  # This selects a callable
            descriptions = [
                selected_template(label)
                for label in dataset_info.get_label_mapping().values()
            ]

            tokens_batch = [self.tokenizer(description) for description in descriptions]

            batch = {
                key: torch.stack([d[key].clone().detach() for d in tokens_batch]).to(
                    pl_module.device
                )
                for key in tokens_batch[0]
            }

            with torch.no_grad():
                processed_example = pl_module.encode(batch, Modalities.TEXT)

            self.all_dataset_info[dataset_index].set_label_embedding(processed_example)

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
            names = batch["dataset_index"]

            # Filter indices where dataset name is part of the metric_name
            matching_indices = [
                i for i, name in enumerate(names) if str(name.item()) in dataset_name
            ]

            if not matching_indices:
                continue

            output_embeddings_filtered = output_embeddings[matching_indices]
            label_index_filtered = label_index[matching_indices]

            target_embeddings_filtered = [
                self.all_dataset_info[str(name.item())].get_label_embedding()
                for i, name in enumerate(names)
                if i in matching_indices
            ]
            target_embeddings_filtered = torch.stack(
                target_embeddings_filtered
            )  # Stack to tensor (N_filtered, num_classes, embedding_dim)

            # Calculate similarities
            predictions_expanded = output_embeddings_filtered.unsqueeze(
                1
            )  # Shape: (N_filtered, 1, embedding_dim)
            logits = functional.cosine_similarity(
                predictions_expanded, target_embeddings_filtered, dim=-1
            )  # Logits

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
