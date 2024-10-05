"""Retrieval Recall@K metric."""

from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed
from hydra_zen import store
from torchmetrics import Metric
from torchmetrics.retrieval.base import _retrieval_aggregate
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_matmul
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.distributed import gather_all_tensors
from tqdm import tqdm


@store(group="modules/metrics", provider="mmlearn")
class RetrievalRecallAtK(Metric):
    """Retrieval Recall@K metric.

    Computes the Recall@K for retrieval tasks. The metric is computed as follows:

    1. Compute the cosine similarity between the query and the database.
    2. For each query, sort the database in decreasing order of similarity.
    3. Compute the Recall@K as the number of true positives among the top K elements.

    Parameters
    ----------
    top_k : int
        The number of top elements to consider for computing the Recall@K.
    reduction : {"mean", "sum", "none", None}, default="sum"
        Specifies the reduction to apply after computing the pairwise cosine similarity
        scores.
    aggregation : {"mean", "median", "min", "max"} or callable, default="mean"
        Specifies the aggregation function to apply to the Recall@K values computed
        in batches. If a callable is provided, it should accept a tensor of values
        and a keyword argument 'dim' and return a single scalar value.
    kwargs : Any
        Additional arguments to be passed to the torchmetrics.Metric class.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    indexes: List[torch.Tensor]
    x: List[torch.Tensor]
    y: List[torch.Tensor]
    num_samples: torch.Tensor

    def __init__(
        self,
        top_k: int,
        reduction: Literal["mean", "sum", "none", None] = "sum",
        aggregation: Union[
            Literal["mean", "median", "min", "max"],
            Callable[[torch.Tensor, int], torch.Tensor],
        ] = "mean",
        **kwargs: Any,
    ) -> None:
        """Initialize the metric."""
        super().__init__(**kwargs)

        if top_k is not None and not (isinstance(top_k, int) and top_k > 0):
            raise ValueError("`top_k` has to be a positive integer or None")
        self.top_k = top_k

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}"
            )
        self.reduction = reduction

        if not (
            aggregation in ("mean", "median", "min", "max") or callable(aggregation)
        ):
            raise ValueError(
                "Argument `aggregation` must be one of `mean`, `median`, `min`, `max` or a custom callable function"
                f"which takes tensor of values, but got {aggregation}."
            )
        self.aggregation = aggregation

        self.add_state("x", default=[], dist_reduce_fx="cat")
        self.add_state("y", default=[], dist_reduce_fx="cat")
        self.add_state("indexes", default=[], dist_reduce_fx="cat")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="cat")

        self._batch_size: int = -1

        self.compute_on_cpu = True
        self.sync_on_compute = False
        self.dist_sync_on_step = False
        self._to_sync = self.sync_on_compute
        self._should_unsync = False

    def _is_distributed(self) -> bool:
        if self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        return distributed_available() if callable(distributed_available) else False  # type: ignore[no-any-return]

    def update(self, x: torch.Tensor, y: torch.Tensor, indexes: torch.Tensor) -> None:
        """Check shape, convert dtypes and add to accumulators.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings (unnormalized) of shape `(N, D)` where `N` is the number
            of samples and `D` is the number of dimensions.
        y : torch.Tensor
            Embeddings (unnormalized) of shape `(M, D)` where `M` is the number
            of samples and `D` is the number of dimensions.
        indexes : torch.Tensor
            Index tensor of shape `(N,)` where `N` is the number of samples.
            This specifies which sample in 'y' is the positive match for each
            sample in 'x'.

        """
        if indexes is None:
            raise ValueError("Argument `indexes` cannot be None")

        x, y, indexes = _update_batch_inputs(x.clone(), y.clone(), indexes.clone())

        # offset batch indexes by the number of samples seen so far
        indexes += self.num_samples

        local_batch_size = torch.zeros_like(self.num_samples) + x.size(0)
        if self._is_distributed():
            x = dim_zero_cat(gather_all_tensors(x, self.process_group))
            y = dim_zero_cat(gather_all_tensors(y, self.process_group))
            indexes = dim_zero_cat(
                gather_all_tensors(indexes.clone(), self.process_group)
            )

            # offset indexes for each device
            bsz_per_device = dim_zero_cat(
                gather_all_tensors(local_batch_size, self.process_group)
            )
            cum_local_bsz = torch.cumsum(bsz_per_device, dim=0)
            for device_idx in range(1, bsz_per_device.numel()):
                indexes[cum_local_bsz[device_idx - 1] : cum_local_bsz[device_idx]] += (
                    cum_local_bsz[device_idx - 1]
                )

            # update the global sample count
            self.num_samples += cum_local_bsz[-1]

            self._is_synced = True
        else:
            self.num_samples += x.size(0)

        self.x.append(x)
        self.y.append(y)
        self.indexes.append(indexes)

        if self._batch_size == -1:
            self._batch_size = x.size(0)  # global batch size

    def compute(self) -> torch.Tensor:
        """Compute the metric in a RAM-efficient manner.

        Returns
        -------
        torch.Tensor
            The computed metric.
        """
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)

        # normalize embeddings
        x_norm = x / x.norm(dim=-1, p=2, keepdim=True)
        y_norm = y / y.norm(dim=-1, p=2, keepdim=True)

        # instantiate reduction function
        reduction_mapping: Dict[
            Optional[str], Callable[[torch.Tensor], torch.Tensor]
        ] = {
            "sum": partial(torch.sum, dim=-1),
            "mean": partial(torch.mean, dim=-1),
            "none": lambda x: x,
            None: lambda x: x,
        }

        # concatenate indexes of true pairs
        indexes = dim_zero_cat(self.indexes)

        results = []
        for start in tqdm(
            range(0, len(x), self._batch_size), desc=f"Recall@{self.top_k}"
        ):
            end = start + self._batch_size
            # compute the cosine similarity
            x_norm_batch = x_norm[start:end]
            similarity = _safe_matmul(x_norm_batch, y_norm)
            scores: torch.Tensor = reduction_mapping[self.reduction](similarity)
            indexes_batch = indexes[start:end]
            positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
            positive_pairs[torch.arange(len(scores)), indexes_batch] = True
            # compute recall_at_k
            result = recall_at_k(scores, positive_pairs, self.top_k)
            results.append(result)

        return _retrieval_aggregate(
            (torch.cat([x.to(scores) for x in results]) > 0).float(), self.aggregation
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward method is not supported."""
        raise NotImplementedError(
            "RetrievalRecallAtK metric does not support forward method"
        )


# modified from:
# https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
def recall_at_k(
    scores: torch.Tensor, positive_pairs: torch.Tensor, k: int
) -> torch.Tensor:
    """Compute the recall at k for each sample.

    Parameters
    ----------
    scores : torch.Tensor
        Compatibility score between embeddings (num_x, num_y).
    positive_pairs : torch.Tensor
        Boolean matrix of positive pairs (num_x, num_y).
    k : int
        Consider only the top k elements for each query.

    Returns
    -------
    recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    return nb_true_positive / nb_positive


def _update_batch_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    indexes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update and returns variables required to compute Retrieval Recall.

    Checks for same shape of input tensors.

    Parameters
    ----------
    x : torch.Tensor
        Predicted tensor.
    y : torch.Tensor
        Ground truth tensor.
    indexes : torch.Tensor
        Index tensor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns updated tensors required to compute Retrieval Recall.

    """
    _check_same_shape(x, y)
    if x.ndim != 2:
        raise ValueError(
            "Expected input to retrieval recall to be 2D tensors of shape `[N,D]`, "
            "where `N` is the number of samples and `D` is the number of dimensions, "
            f"but got tensor of shape {x.shape}"
        )
    if not indexes.numel() or not indexes.size():
        raise ValueError(
            "`indexes`, `x` and `y` must be non-empty and non-scalar tensors",
        )

    x = x.float()
    y = y.float()
    indexes = indexes.long()

    return x, y, indexes
