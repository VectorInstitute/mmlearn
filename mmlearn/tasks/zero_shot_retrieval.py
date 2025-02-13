"""Zero-shot cross-modal retrieval evaluation task."""

from dataclasses import dataclass
from typing import Any, Optional

import lightning.pytorch as pl
import torch
import torch.distributed
import torch.distributed.nn
from hydra_zen import store
from torchmetrics import MetricCollection

from mmlearn.datasets.core import Modalities
from mmlearn.modules.metrics import RetrievalRecallAtK
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class RetrievalTaskSpec:
    """Specification for a retrieval task."""

    #: The query modality.
    query_modality: str

    #: The target modality.
    target_modality: str

    #: The top-k values for which to compute the retrieval recall metrics.
    top_k: list[int]


@store(group="eval_task", provider="mmlearn")
class ZeroShotCrossModalRetrieval(EvaluationHooks):
    """Zero-shot cross-modal retrieval evaluation task.

    This task evaluates the retrieval performance of a model on a set of query-target
    pairs. The model is expected to produce embeddings for both the query and target
    modalities. The task computes the retrieval recall at `k` for each pair of
    modalities.

    Parameters
    ----------
    task_specs : list[RetrievalTaskSpec]
        A list of retrieval task specifications. Each specification defines the query
        and target modalities, as well as the top-k values for which to compute the
        retrieval recall metrics.

    """

    def __init__(self, task_specs: list[RetrievalTaskSpec]) -> None:
        super().__init__()

        self.task_specs = task_specs
        self.metrics: dict[tuple[str, str], MetricCollection] = {}
        self._available_modalities = set()

        for spec in self.task_specs:
            query_modality = spec.query_modality
            target_modality = spec.target_modality
            assert Modalities.has_modality(query_modality)
            assert Modalities.has_modality(target_modality)

            self.metrics[(query_modality, target_modality)] = MetricCollection(
                {
                    f"{query_modality}_to_{target_modality}_R@{k}": RetrievalRecallAtK(
                        top_k=k, aggregation="mean", reduction="none"
                    )
                    for k in spec.top_k
                }
            )
            self._available_modalities.add(query_modality)
            self._available_modalities.add(target_modality)

    def on_evaluation_epoch_start(self, pl_module: pl.LightningModule) -> None:
        """Move the metrics to the device of the Lightning module."""
        self.metrics.to(pl_module.device)  # type: ignore[attr-defined]

    def evaluation_step(
        self,
        pl_module: pl.LightningModule,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Run the forward pass and update retrieval recall metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        batch : dict[str, torch.Tensor]
            A dictionary of batched input tensors.
        batch_idx : int
            The index of the batch.

        """
        if pl_module.trainer.sanity_checking:
            return

        outputs: dict[str, Any] = {}
        for modality_name in self._available_modalities:
            if modality_name in batch:
                outputs[modality_name] = pl_module.encode(
                    batch, Modalities.get_modality(modality_name), normalize=False
                )
        for (query_modality, target_modality), metric in self.metrics.items():
            if query_modality not in outputs or target_modality not in outputs:
                continue
            query_embeddings: torch.Tensor = outputs[query_modality]
            target_embeddings: torch.Tensor = outputs[target_modality]
            indexes = torch.arange(query_embeddings.size(0), device=pl_module.device)

            metric.update(query_embeddings, target_embeddings, indexes)

    def on_evaluation_epoch_end(
        self, pl_module: pl.LightningModule
    ) -> Optional[dict[str, Any]]:
        """Compute the retrieval recall metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.

        Returns
        -------
        Optional[dict[str, Any]]
            A dictionary of evaluation results or `None` if no results are available.
        """
        if pl_module.trainer.sanity_checking:
            return None

        results = {}
        for metric in self.metrics.values():
            results.update(metric.compute())
            metric.reset()

        eval_type = "val" if pl_module.trainer.validating else "test"

        for key, value in results.items():
            pl_module.log(f"{eval_type}/{key}", value)

        return results
