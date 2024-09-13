"""Implementations of the contrastive loss and its variants."""

from typing import Dict, Tuple

import torch
import torch.distributed as dist
from hydra_zen import store
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchmetrics.utilities.compute import _safe_matmul
from torchmetrics.utilities.distributed import gather_all_tensors


@store(group="modules/losses", provider="mmlearn")
class CLIPLoss(nn.Module):
    """CLIP Loss module.

    Parameters
    ----------
    l2_normalize : bool, default=False
        Whether to L2 normalize the features.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e. `local_features@global_features`.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    cache_labels : bool, default=False
        Whether to cache the labels.

    """

    def __init__(
        self,
        l2_normalize: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
    ):
        """Initialize the loss."""
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.l2_normalize = l2_normalize

        # cache state
        self._prev_num_logits = 0
        self._labels: Dict[torch.device, torch.Tensor] = {}

    def _get_ground_truth(
        self, device: torch.device, num_logits: int, rank: int, world_size: int
    ) -> torch.Tensor:
        """Return the ground-truth labels.

        Parameters
        ----------
        device : torch.device
            Device to store the labels.
        num_logits : int
            Number of logits.
        rank : int
            Rank of the current process.
        world_size : int
            Number of processes.

        Returns
        -------
        torch.Tensor
            Ground-truth labels.
        """
        # calculate ground-truth and cache if enabled
        if self._prev_num_logits != num_logits or device not in self._labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self._labels[device] = labels
                self._prev_num_logits = num_logits
        else:
            labels = self._labels[device]
        return labels

    def _get_logits(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the logits.

        Parameters
        ----------
        features_1 : torch.Tensor
            First feature tensor.
        features_2 : torch.Tensor
            Second feature tensor
        rank : int
            Rank of the current process.
        world_size : int
            Number of processes.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Logits per feature_1 and feature_2, respectively.

        """
        if world_size > 1:
            all_features_1 = gather_features(
                features_1, self.local_loss, self.gather_with_grad, rank
            )
            all_features_2 = gather_features(
                features_2, self.local_loss, self.gather_with_grad, rank
            )

            if self.local_loss:
                logits_per_feature_1 = _safe_matmul(features_1, all_features_2)
                logits_per_feature_2 = _safe_matmul(features_2, all_features_1)
            else:
                logits_per_feature_1 = _safe_matmul(all_features_1, all_features_2)
                logits_per_feature_2 = logits_per_feature_1.T
        else:
            logits_per_feature_1 = _safe_matmul(features_1, features_2)
            logits_per_feature_2 = _safe_matmul(features_2, features_1)

        return logits_per_feature_1, logits_per_feature_2

    def forward(
        self, features_1: torch.Tensor, features_2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the CLIP-style loss between two sets of features.

        Parameters
        ----------
        features_1 : torch.Tensor
            First set of features.
        features_2 : torch.Tensor
            Second set of features.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if world_size > 1 else 0

        if self.l2_normalize:
            features_1 = F.normalize(features_1, p=2, dim=-1)
            features_2 = F.normalize(features_2, p=2, dim=-1)

        logits_per_feat1, logits_per_feat2 = self._get_logits(
            features_1, features_2, rank=rank, world_size=world_size
        )
        labels = self._get_ground_truth(
            features_1.device,
            logits_per_feat1.shape[0],
            rank=rank,
            world_size=world_size,
        )

        return (
            F.cross_entropy(logits_per_feat1, labels)
            + F.cross_entropy(logits_per_feat2, labels)
        ) / 2


def gather_features(
    features: torch.Tensor,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
) -> torch.Tensor:
    """Gather features across all processes.

    Parameters
    ----------
    features : torch.Tensor
        First feature tensor to gather.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e.
        `matmul(local_features, global_features)`. If False, this method ensures
        that the gathered features contain local features for the current rank.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    rank : int, default=0
        Rank of the current process.

    Returns
    -------
    torch.Tensor
        Gathered features.
    """
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = gather_all_tensors(features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)

    return all_features
