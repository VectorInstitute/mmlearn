"""Zero-shot and linear probing classification evaluation task."""

import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchmetrics
from hydra_zen import store
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional
from torchmetrics import Metric, MetricCollection

from mmlearn.constants import MEDICAL_TEMPLATES
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""

    metric_name: str
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
        self.metrics: Dict[Any, Metric] = {}

        self.tokenizer = tokenizer

    def on_evaluation_epoch_start(
        self, pl_module: LightningModule, all_dataset_info: Dict[str, Any]
    ) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer must be set in the dataset to generate tokenized label descriptions"
            )

        # Set the label embeddings
        for name, dataset_info in all_dataset_info.items():
            descriptions = [
                random.choice(MEDICAL_TEMPLATES)(label)
                for label in dataset_info.get_label_mapping().values()
            ]

            processed_descriptions = []
            for description in descriptions:
                batch = {}
                tokens = self.tokenizer(description)

                batch.update(
                    {
                        key: value.unsqueeze(0).to(pl_module.device)
                        if torch.is_tensor(value)
                        else value
                        for key, value in tokens.items()
                    }
                )
                batch[Modalities.RGB] = torch.rand(1, 3, 224, 224).to(pl_module.device)

                with torch.no_grad():
                    processed_example = pl_module(batch)
                    embedding = processed_example[
                        Modalities.get_modality(Modalities.TEXT).embedding
                    ]
                    processed_descriptions.append(embedding)

            all_dataset_info[name].set_label_embedding(
                torch.stack(processed_descriptions).squeeze(1)
            )

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
