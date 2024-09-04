"""Zero-shot cross-modal retrieval evaluation task."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.distributed
import torch.distributed.nn
from hydra_zen import store
from torchmetrics import MetricCollection

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.modules.metrics import RetrievalRecallAtK
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class RetrievalTaskSpec:
    """Specification for a retrieval task."""

    query_modality: str
    target_modality: str
    top_k: List[int]


@store(group="eval_task", provider="mmlearn")
class ZeroShotCrossModalRetrieval(EvaluationHooks):
    """Zero-shot cross-modal retrieval evaluation task.

    This task evaluates the retrieval performance of a model on a set of query-target
    pairs. The model is expected to produce embeddings for both the query and target
    modalities. The task computes the retrieval recall at `k` for each pair of
    modalities.

    Parameters
    ----------
    task_specs : List[RetrievalTaskSpec]
        A list of retrieval task specifications. Each specification defines the query
        and target modalities, as well as the top-k values for which to compute the
        retrieval recall metrics.

    """

    def __init__(self, task_specs: List[RetrievalTaskSpec]):
        """Initialize the module."""
        super().__init__()

        self.task_specs = task_specs
        self.metrics: Dict[Tuple[Modality, Modality], MetricCollection] = {}

        for spec in self.task_specs:
            assert Modalities.has_modality(spec.query_modality)
            assert Modalities.has_modality(spec.target_modality)

            query_modality = Modalities.get_modality(spec.query_modality)
            target_modality = Modalities.get_modality(spec.target_modality)

            self.metrics[(query_modality, target_modality)] = MetricCollection(
                {
                    f"{query_modality}_to_{target_modality}_R@{k}": RetrievalRecallAtK(
                        top_k=k, aggregation="mean", reduction="none"
                    )
                    for k in spec.top_k
                }
            )

    def on_evaluation_epoch_start(self, pl_module: pl.LightningModule) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)

    def evaluation_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Run the forward pass and update retrieval recall metrics.

        Parameters
        ----------
        trainer : pl.Trainer
            A reference to the Lightning Trainer.
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        batch : Dict[str, torch.Tensor]
            A dictionary of batched input tensors.
        batch_idx : int
            The index of the batch.

        """
        if trainer.sanity_checking:
            return

        outputs: Dict[Union[str, Modality], Any] = pl_module(batch)
        for (query_modality, target_modality), metric in self.metrics.items():
            query_embeddings: torch.Tensor = outputs[query_modality.embedding]
            target_embeddings: torch.Tensor = outputs[target_modality.embedding]
            indexes = torch.arange(query_embeddings.size(0), device=pl_module.device)

            metric.update(query_embeddings, target_embeddings, indexes)

    def on_evaluation_epoch_end(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """Compute the retrieval recall metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        results = {}
        for metric in self.metrics.values():
            results.update(metric.compute())
            metric.reset()

        return results