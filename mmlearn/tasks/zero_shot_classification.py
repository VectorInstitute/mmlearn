"""Zero-shot classification evaluation task."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

import torch
from hydra_zen import store
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import move_data_to_device
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.utilities.compute import _safe_matmul
from tqdm.auto import tqdm

from mmlearn.datasets.core import CombinedDataset, Modalities
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""

    query_modality: str
    top_k: List[int]


@store(group="eval_task", provider="mmlearn")
class ZeroShotClassification(EvaluationHooks):
    """Zero-shot classification evaluation task.

    This task evaluates the zero-shot classification performance.

    Parameters
    ----------
    task_specs : List[ClassificationTaskSpec]
        A list of classification task specifications.
    tokenizer : Callable[[Union[str, list[str]]], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        A function to tokenize text inputs.
    """  # noqa: W505

    def __init__(
        self,
        task_specs: List[ClassificationTaskSpec],
        tokenizer: Callable[
            [Union[str, list[str]]], Union[torch.Tensor, Dict[str, torch.Tensor]]
        ],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.task_specs = task_specs
        for spec in self.task_specs:
            assert Modalities.has_modality(spec.query_modality)

        self.metrics: Dict[tuple[str, int], MetricCollection] = {}
        self._embeddings_store: Dict[int, torch.Tensor] = {}

    def on_evaluation_epoch_start(self, pl_module: LightningModule) -> None:
        """Set up the evaluation task.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        if pl_module.trainer.validating:
            eval_dataset: CombinedDataset = pl_module.trainer.val_dataloaders.dataset
        elif pl_module.trainer.testing:
            eval_dataset = pl_module.trainer.test_dataloaders.dataset
        else:
            raise ValueError(
                "ZeroShotClassification task is only supported for validation and testing."
            )

        self.all_dataset_info = {}

        # create metrics for each dataset/query_modality combination
        if not self.metrics:
            for dataset_index, dataset in enumerate(eval_dataset.datasets):
                dataset_name = getattr(dataset, "name", dataset.__class__.__name__)
                try:
                    id2label: dict[int, str] = dataset.id2label
                except AttributeError:
                    raise ValueError(
                        f"Dataset '{dataset_name}' must have a `id2label` attribute "
                        "to perform zero-shot classification."
                    ) from None

                try:
                    zero_shot_prompt_templates: list[str] = (
                        dataset.zero_shot_prompt_templates
                    )
                except AttributeError:
                    raise ValueError(
                        "Dataset must have a `zero_shot_prompt_templates` attribute to perform zero-shot classification."
                    ) from None

                num_classes = len(id2label)

                self.all_dataset_info[dataset_index] = {
                    "name": dataset_name,
                    "id2label": id2label,
                    "prompt_templates": zero_shot_prompt_templates,
                    "num_classes": num_classes,
                }

                for spec in self.task_specs:
                    query_modality = Modalities.get_modality(spec.query_modality).name
                    self.metrics[(query_modality, dataset_index)] = (
                        self._create_metrics(
                            num_classes,
                            spec.top_k,
                            prefix=f"{dataset_name}/{query_modality}_",
                            postfix="",
                        )
                    )

        for metric in self.metrics.values():
            metric.to(pl_module.device)

        for dataset_index, dataset_info in self.all_dataset_info.items():
            id2label = dataset_info["id2label"]
            prompt_templates: list[str] = dataset_info["prompt_templates"]
            labels = list(id2label.values())

            with torch.no_grad():
                chunk_size = 10
                all_embeddings = []

                for i in tqdm(
                    range(0, len(labels), chunk_size),
                    desc="Encoding class descriptions",
                ):
                    batch_labels = labels[i : min(i + chunk_size, len(labels))]
                    descriptions = [
                        template.format(label)
                        for label in batch_labels
                        for template in prompt_templates
                    ]
                    tokenized_descriptions = move_data_to_device(
                        self.tokenizer(descriptions),
                        pl_module.device,
                    )

                    # Encode the chunk using the pl_module's encode method
                    chunk_embeddings = pl_module.encode(
                        tokenized_descriptions, Modalities.TEXT
                    )  # shape: [chunk_size x len(prompt_templates), embed_dim]
                    chunk_embeddings /= chunk_embeddings.norm(p=2, dim=-1, keepdim=True)
                    chunk_embeddings = chunk_embeddings.reshape(
                        len(batch_labels), len(prompt_templates), -1
                    ).mean(dim=1)  # shape: [chunk_size, embed_dim]
                    chunk_embeddings /= chunk_embeddings.norm(p=2, dim=-1, keepdim=True)

                    # Append the chunk embeddings to the list
                    all_embeddings.append(chunk_embeddings)

                # Concatenate all chunk embeddings into a single tensor
                class_embeddings = torch.cat(all_embeddings, dim=0)

            self._embeddings_store[dataset_index] = class_embeddings

    def evaluation_step(
        self, pl_module: LightningModule, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute logits and update metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        batch : Dict[str, torch.Tensor]
            A batch of data.
        batch_idx : int
            The index of the batch.
        """
        if pl_module.trainer.sanity_checking:
            return

        for (query_modality, dataset_index), metric_collection in self.metrics.items():
            matching_indices = torch.where(batch["dataset_index"] == dataset_index)[0]

            if not matching_indices.numel():
                continue

            class_embeddings = self._embeddings_store[dataset_index]
            query_embeddings: torch.Tensor = pl_module.encode(
                batch, Modalities.get_modality(query_modality)
            )
            query_embeddings /= query_embeddings.norm(p=2, dim=-1, keepdim=True)
            query_embeddings = query_embeddings[matching_indices]

            if self.all_dataset_info[dataset_index]["num_classes"] == 2:
                softmax_output = _safe_matmul(
                    query_embeddings, class_embeddings
                ).softmax(dim=-1)
                logits = softmax_output[:, 1] - softmax_output[:, 0]
            else:
                logits = 100.0 * _safe_matmul(query_embeddings, class_embeddings)
            targets = batch[Modalities.get_modality(query_modality).target][
                matching_indices
            ]

            metric_collection.update(logits, targets)

    def on_evaluation_epoch_end(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Compute and reset metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.

        Returns
        -------
        Dict[str, Any]
            The computed metrics.
        """
        results = {}
        for metric_collection in self.metrics.values():
            results.update(metric_collection.compute())
            metric_collection.reset()

        self._embeddings_store.clear()

        eval_type = "val" if pl_module.trainer.validating else "test"
        for key, value in results.items():
            pl_module.log(f"{eval_type}/{key}", value)

        return results

    @staticmethod
    def _create_metrics(
        num_classes: int, top_k: List[int], prefix: str, postfix: str
    ) -> MetricCollection:
        """Create a collection of classification metrics."""
        task_type = "binary" if num_classes == 2 else "multiclass"
        acc_metrics = (
            {
                f"top{k}_accuracy": Accuracy(
                    task=task_type, num_classes=num_classes, top_k=k, average="micro"
                )
                for k in top_k
            }
            if num_classes > 2
            else {"accuracy": Accuracy(task=task_type, num_classes=num_classes)}
        )
        return MetricCollection(
            {
                "precision": Precision(
                    task=task_type,
                    num_classes=num_classes,
                    average="macro" if num_classes > 2 else "micro",
                ),
                "recall": Recall(
                    task=task_type,
                    num_classes=num_classes,
                    average="macro" if num_classes > 2 else "micro",
                ),
                "f1_score_macro": F1Score(
                    task=task_type,
                    num_classes=num_classes,
                    average="macro" if num_classes > 2 else "micro",
                ),
                "aucroc": AUROC(task=task_type, num_classes=num_classes),
                **acc_metrics,
            },
            prefix=prefix,
            postfix=postfix,
            compute_groups=True,
        )
