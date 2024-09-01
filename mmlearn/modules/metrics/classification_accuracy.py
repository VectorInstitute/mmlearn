from hydra_zen import store
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors
from typing import Any, Callable, Literal, Optional, Union, List, Tuple
import torch

from torchmetrics.utilities.data import dim_zero_cat

@store(group="modules/metrics", provider="mmlearn")
class ZeroShotClassificationAccuracy(Metric):
    """
    Zero-Shot Classification Accuracy metric.

    Computes accuracy for zero-shot classification tasks where the model predicts
    labels that it has not seen during training.

    Parameters
    ----------
    num_classes : int
        The number of classes to consider in the zero-shot classification task.
    reduction : {"mean", "sum", "none", None}, default="mean"
        Specifies the reduction to apply to the accuracy scores.
    aggregation : {"mean", "median", "min", "max"} or callable, default="mean"
        Specifies the aggregation function to apply to the accuracy scores computed
        in batches. If a callable is provided, it should accept a tensor of values
        and a keyword argument 'dim' and return a single scalar value.
    kwargs : Any
        Additional arguments to be passed to the torchmetrics.Metric class.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        top_k: Tuple[int, ...] = (1,),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("true_indices", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, true_indices: torch.Tensor) -> None:
        """
        Accumulate predictions and true label indices.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted embeddings of shape `(N, embedding_dim)` where `N` is the number of samples.
        true_indices : torch.Tensor
            True label indices of shape `(N,)` where `N` is the number of samples.
        """
        self.predictions.append(preds)
        self.true_indices.append(true_indices)
        
        
    def _is_distributed(self) -> bool:
        if self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        return distributed_available() if callable(distributed_available) else False  # type: ignore[no-any-return]


    def compute(self, target_embeddings: torch.Tensor) -> List[float]:
        """
        Compute the top-k accuracies from the embeddings using dynamically provided target embeddings.

        Parameters
        ----------
        target_embeddings : torch.Tensor
            Target embeddings for all labels (shape `[num_classes, embedding_dim]`).

        Returns
        -------
        List[float]
            List of computed top-k accuracies.
        """
        predictions = torch.cat(self.predictions, dim=0)
        true_indices = torch.cat(self.true_indices, dim=0)

        if self._is_distributed():
            predictions = torch.cat(gather_all_tensors(predictions), dim=0)
            true_indices = torch.cat(gather_all_tensors(true_indices), dim=0)
        return self.zero_shot_accuracy(predictions, true_indices, target_embeddings)

    def zero_shot_accuracy(self, predictions, true_indices, target_embeddings):
        """
        Compute zero-shot accuracy using cosine similarity.

        predictions: torch.Tensor
            Predicted embeddings, shape (N, embedding_dim).
        true_indices: torch.Tensor
            True label indices, shape (N,).
        target_embeddings: torch.Tensor
            Precomputed target embeddings for all labels, shape (num_classes, embedding_dim).

        Returns
        -------
        list of top-k accuracies in the same order as `top_k`
        """
        similarities = torch.matmul(predictions, target_embeddings.t())
        top_k_values = similarities.topk(max(self.top_k), dim=1).indices
        correct = top_k_values == true_indices.unsqueeze(1)
        accuracies = [correct[:, :k].any(dim=1).float().mean().item() for k in self.top_k]
        return torch.tensor(accuracies).mean()
    