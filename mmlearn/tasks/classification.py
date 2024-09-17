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

from mmlearn.constants import NAME_KEY
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
class Classification(EvaluationHooks):
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
        mode: Literal["zeroshot", "linear_probing"] = "zeroshot",
    ):
        super().__init__()
        self.mode = mode
        self.task_specs = task_specs
        self.metrics: Dict[Any, Metric] = {}

        self.tokenizer = tokenizer

    def on_evaluation_epoch_start(
        self, pl_module: LightningModule, all_dataset_info: Dict[str, Any]
    ) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)

        if self.mode == "zeroshot":

            class LabelDescriptionDataset(Dataset):
                def __init__(
                    self, descriptions: List[str], tokenizer: PreTrainedTokenizerBase
                ) -> None:
                    self.descriptions = descriptions
                    self.tokenizer = tokenizer

                def __len__(self) -> int:
                    return len(self.descriptions)

                def __getitem__(self, idx: int) -> Example:
                    description = self.descriptions[idx]
                    tokens = self.tokenizer(description)

                    example = Example(
                        {
                            Modalities.RGB: torch.rand(3, 224, 224),
                            Modalities.TEXT: description,
                        }
                    )

                    if tokens is not None:
                        if isinstance(tokens, dict):  # output of HFTokenizer
                            assert (
                                Modalities.TEXT in tokens
                            ), f"Missing key `{Modalities.TEXT}` in tokens."
                            example.update(tokens)
                        else:
                            example[Modalities.TEXT] = tokens

                    return example

            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer must be set in the dataset to generate tokenized label descriptions"
                )

            # Set the label embeddings
            for name, dataset_info in all_dataset_info.items():
                descriptions = [
                    "This image has a sign of " + label
                    for label in dataset_info.get_label_mapping().values()
                ]

                dataset = LabelDescriptionDataset(descriptions, self.tokenizer)
                batch_size = len(dataset)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_example_list,
                )
                batch = next(iter(dataloader))
                batch = {
                    key: value.to(pl_module.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                all_dataset_info[name].set_label_embedding(
                    pl_module(batch)[Modalities.get_modality(Modalities.TEXT).embedding]
                )

        self.all_dataset_info = all_dataset_info

        for dataset_name in list(all_dataset_info.keys()):
            for spec in self.task_specs:
                assert Modalities.has_modality(spec.query_modality)
                query_modality = Modalities.get_modality(spec.query_modality)
                mode = self.mode
                metric_name = spec.metric_name
                metric = getattr(torchmetrics, metric_name)
                num_classes = all_dataset_info[dataset_name].get_class_count()
                task = "multiclass" if num_classes > 2 else "binary"
                self.metrics.update(
                    {
                        (
                            query_modality,
                            mode,
                            metric_name,
                            dataset_name,
                        ): MetricCollection(
                            {
                                f"{query_modality}_{mode}_C_{metric_name}@{k}_{dataset_name}": (
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
                                    ).to("cuda")
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
            _,
            dataset_name,
        ), metric in self.metrics.items():
            output_embeddings = outputs[query_modality.embedding]  # Predictions
            label_index = batch[query_modality.target]  # True labels
            names = batch[
                NAME_KEY
            ]  # List of dataset names for each element in the batch

            # Filter indices where dataset name is part of the metric_name
            matching_indices = [
                i for i, name in enumerate(names) if name in dataset_name
            ]

            if not matching_indices:
                continue

            output_embeddings_filtered = output_embeddings[matching_indices]
            label_index_filtered = label_index[matching_indices]

            if self.mode == "zeroshot":
                target_embeddings_filtered = [
                    self.all_dataset_info[name].get_label_embedding()
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

            elif self.mode == "linear_probing":
                logits = output_embeddings_filtered

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
            _,
        ), metric in self.metrics.items():
            results.update(metric.compute())
            metric.reset()
        return results
