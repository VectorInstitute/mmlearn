"""Implementations of the contrastive loss and its variants."""

import itertools
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from hydra_zen import store
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchmetrics.utilities.compute import _safe_matmul

from mmlearn.datasets.core import find_matching_indices
from mmlearn.datasets.core.modalities import Modalities
from mmlearn.tasks.contrastive_pretraining import LossPairSpec


@store(group="modules/losses", provider="mmlearn")
class ContrastiveLoss(nn.Module):
    """Contrastive Loss module.

    Parameters
    ----------
    l2_normalize : bool, default=False
        Whether to L2 normalize the features.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e. `local_features@global_features`.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    modality_alignment : bool, default=False
        Whether to include modality alignment loss. This loss considers all features
        from the same modality as positive pairs and all features from different
        modalities as negative pairs.
    cache_labels : bool, default=False
        Whether to cache the labels.

    """

    def __init__(
        self,
        l2_normalize: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        modality_alignment: bool = False,
        cache_labels: bool = False,
    ):
        """Initialize the loss."""
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.l2_normalize = l2_normalize
        self.modality_alignment = modality_alignment

        # cache state
        self._prev_num_logits = 0
        self._labels: Dict[torch.device, torch.Tensor] = {}

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        example_ids: dict[str, torch.Tensor],
        logit_scale: torch.Tensor,
        modality_loss_pairs: list[LossPairSpec],
    ) -> torch.Tensor:
        """Calculate the contrastive loss.

        Parameters
        ----------
        embeddings : dict[str, torch.Tensor]
            Dictionary of embeddings, where the key is the modality name and the value
            is the corresponding embedding tensor.
        example_ids : dict[str, torch.Tensor]
            Dictionary of example IDs, where the key is the modality name and the value
            is a tensor tuple of the dataset index and the example index.
        logit_scale : torch.Tensor
            Scale factor for the logits.
        modality_loss_pairs : List[LossPairSpec]
            Specification of the modality pairs for which the loss should be calculated.

        Returns
        -------
        torch.Tensor
            Contrastive loss.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if world_size > 1 else 0

        if self.l2_normalize:
            embeddings = {k: F.normalize(v, p=2, dim=-1) for k, v in embeddings.items()}

        if world_size > 1:  # gather embeddings and example_ids across all processes
            # NOTE: gathering dictionaries of tensors across all processes
            # (keys + values, as opposed to just values) is especially important
            # for the modality_alignment loss, which requires all embeddings
            all_embeddings = _gather_dicts(
                embeddings,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=rank,
            )
            all_example_ids = _gather_dicts(
                example_ids,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=rank,
            )
        else:
            all_embeddings = embeddings
            all_example_ids = example_ids

        losses = []
        for loss_pairs in modality_loss_pairs:
            logits_per_feature_a, logits_per_feature_b, skip_flag = self._get_logits(
                loss_pairs.modalities,
                per_device_embeddings=embeddings,
                all_embeddings=all_embeddings,
                per_device_example_ids=example_ids,
                all_example_ids=all_example_ids,
                logit_scale=logit_scale,
                world_size=world_size,
            )
            if logits_per_feature_a is None or logits_per_feature_b is None:
                continue

            labels = self._get_ground_truth(
                logits_per_feature_a.shape,
                device=logits_per_feature_a.device,
                rank=rank,
                world_size=world_size,
                skipped_process=skip_flag,
            )

            if labels.numel() != 0:
                losses.append(
                    (
                        (
                            F.cross_entropy(logits_per_feature_a, labels)
                            + F.cross_entropy(logits_per_feature_b, labels)
                        )
                        / 2
                    )
                    * loss_pairs.weight
                )

        if self.modality_alignment:
            losses.append(
                self._compute_modality_alignment_loss(all_embeddings, logit_scale)
            )

        if not losses:  # no loss to compute (e.g. no paired data in batch)
            losses.append(
                torch.tensor(
                    0.0,
                    device=logit_scale.device,
                    dtype=next(iter(embeddings.values())).dtype,
                )
            )

        return torch.stack(losses).sum()

    def _get_ground_truth(
        self,
        logits_shape: tuple[int, int],
        device: torch.device,
        rank: int,
        world_size: int,
        skipped_process: bool,
    ) -> torch.Tensor:
        """Return the ground-truth labels.

        Parameters
        ----------
        logits_shape : tuple[int, int]
            Shape of the logits tensor.
        device : torch.device
            Device on which the labels should be created.
        rank : int
            Rank of the current process.
        world_size : int
            Number of processes.
        skipped_process : bool
            Whether the current process skipped the computation due to lack of data.

        Returns
        -------
        torch.Tensor
            Ground-truth labels.
        """
        num_logits = logits_shape[-1]

        # calculate ground-truth and cache if enabled
        if self._prev_num_logits != num_logits or device not in self._labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)

            if world_size > 1 and self.local_loss:
                local_size = torch.tensor(
                    0 if skipped_process else logits_shape[0], device=device
                )
                # NOTE: all processes must participate in the all_gather operation
                # even if they have no data to contribute.
                sizes = torch.stack(
                    _simple_gather_all_tensors(
                        local_size, group=dist.group.WORLD, world_size=world_size
                    )
                )
                sizes = torch.cat(
                    [torch.tensor([0], device=sizes.device), torch.cumsum(sizes, dim=0)]
                )
                labels = labels[
                    sizes[rank] : sizes[rank + 1] if rank + 1 < world_size else None
                ]

            if self.cache_labels:
                self._labels[device] = labels
                self._prev_num_logits = num_logits
        else:
            labels = self._labels[device]
        return labels

    def _get_logits(  # noqa: PLR0912
        self,
        modalities: tuple[str, str],
        per_device_embeddings: dict[str, torch.Tensor],
        all_embeddings: dict[str, torch.Tensor],
        per_device_example_ids: dict[str, torch.Tensor],
        all_example_ids: dict[str, torch.Tensor],
        logit_scale: torch.Tensor,
        world_size: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """Calculate the logits for the given modalities.

        Parameters
        ----------
        modalities : tuple[str, str]
            Tuple of modality names.
        per_device_embeddings : dict[str, torch.Tensor]
            Dictionary of embeddings, where the key is the modality name and the value
            is the corresponding embedding tensor.
        all_embeddings : dict[str, torch.Tensor]
            Dictionary of embeddings, where the key is the modality name and the value
            is the corresponding embedding tensor. In distributed mode, this contains
            embeddings from all processes.
        per_device_example_ids : dict[str, torch.Tensor]
            Dictionary of example IDs, where the key is the modality name and the value
            is a tensor tuple of the dataset index and the example index.
        all_example_ids : dict[str, torch.Tensor]
            Dictionary of example IDs, where the key is the modality name and the value
            is a tensor tuple of the dataset index and the example index. In distributed
            mode, this contains example IDs from all processes.
        logit_scale : torch.Tensor
            Scale factor for the logits.
        world_size : int
            Number of processes.

        Returns
        -------
        tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]
            Tuple of logits for the given modalities. If embeddings for the given
            modalities are not available, returns `None` for the logits. The last
            element is a flag indicating whether the process skipped the computation
            due to lack of data.
        """
        modality_a = Modalities.get_modality(modalities[0])
        modality_b = Modalities.get_modality(modalities[1])
        skip_flag = False

        if self.local_loss or world_size == 1:
            if not (
                modality_a.embedding in per_device_embeddings
                and modality_b.embedding in per_device_embeddings
            ):
                if world_size > 1:  # NOTE: not all processes exit here, hence skip_flag
                    skip_flag = True
                else:
                    return None, None, skip_flag

            if not skip_flag:
                indices_a, indices_b = find_matching_indices(
                    per_device_example_ids[modality_a.name],
                    per_device_example_ids[modality_b.name],
                )
                if indices_a.numel() == 0 or indices_b.numel() == 0:
                    if world_size > 1:  # not all processes exit here
                        skip_flag = True
                    else:
                        return None, None, skip_flag

            if not skip_flag:
                features_a = per_device_embeddings[modality_a.embedding][indices_a]
                features_b = per_device_embeddings[modality_b.embedding][indices_b]
            else:
                # all processes must participate in the all_gather operation
                # that follows, even if they have no data to contribute. So,
                # we create empty tensors here.
                features_a = torch.empty(
                    0, device=list(per_device_embeddings.values())[0].device
                )
                features_b = torch.empty(
                    0, device=list(per_device_embeddings.values())[0].device
                )

        if world_size > 1:
            if not (
                modality_a.embedding in all_embeddings
                and modality_b.embedding in all_embeddings
            ):  # all processes exit here
                return None, None, skip_flag

            indices_a, indices_b = find_matching_indices(
                all_example_ids[modality_a.name],
                all_example_ids[modality_b.name],
            )
            if indices_a.numel() == 0 or indices_b.numel() == 0:
                # all processes exit here
                return None, None, skip_flag

            all_features_a = all_embeddings[modality_a.embedding][indices_a]
            all_features_b = all_embeddings[modality_b.embedding][indices_b]

            if self.local_loss:
                if features_a.numel() == 0:
                    features_a = all_features_a
                if features_b.numel() == 0:
                    features_b = all_features_b

                logits_per_feature_a = logit_scale * _safe_matmul(
                    features_a, all_features_b
                )
                logits_per_feature_b = logit_scale * _safe_matmul(
                    features_b, all_features_a
                )
            else:
                logits_per_feature_a = logit_scale * _safe_matmul(
                    all_features_a, all_features_b
                )
                logits_per_feature_b = logits_per_feature_a.T
        else:
            logits_per_feature_a = logit_scale * _safe_matmul(features_a, features_b)
            logits_per_feature_b = logit_scale * _safe_matmul(features_b, features_a)

        return logits_per_feature_a, logits_per_feature_b, skip_flag

    def _compute_modality_alignment_loss(
        self, all_embeddings: dict[str, torch.Tensor], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute the modality alignment loss.

        This loss considers all features from the same modality as positive pairs
        and all features from different modalities as negative pairs.

        Parameters
        ----------
        all_embeddings : dict[str, torch.Tensor]
            Dictionary of embeddings, where the key is the modality name and the value
            is the corresponding embedding tensor.
        logit_scale : torch.Tensor
            Scale factor for the logits.

        Returns
        -------
        torch.Tensor
            Modality alignment loss.

        Notes
        -----
        This loss does not support `local_loss=True`.
        """
        available_modalities = list(all_embeddings.keys())
        # TODO: support local_loss for modality_alignment?
        # if world_size == 1, all_embeddings == embeddings
        all_features = torch.cat(list(all_embeddings.values()), dim=0)

        positive_indices = torch.tensor(
            [
                (i, j)
                if idx == 0
                else (
                    i + all_embeddings[available_modalities[idx - 1]].size(0),
                    j + all_embeddings[available_modalities[idx - 1]].size(0),
                )
                for idx, k in enumerate(all_embeddings)
                for i, j in itertools.combinations(range(all_embeddings[k].size(0)), 2)
            ],
            device=all_features.device,
        )
        logits = logit_scale * _safe_matmul(all_features, all_features)

        target = torch.eye(all_features.size(0), device=all_features.device)
        target[positive_indices[:, 0], positive_indices[:, 1]] = 1

        modality_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )

        target_pos = target.bool()
        target_neg = ~target_pos

        # loss_pos and loss_neg below contain non-zero values only for those
        # elements that are positive pairs and negative pairs respectively.
        loss_pos = torch.zeros(
            logits.size(0), logits.size(0), device=target.device
        ).masked_scatter(target_pos, modality_loss[target_pos])
        loss_neg = torch.zeros(
            logits.size(0), logits.size(0), device=target.device
        ).masked_scatter(target_neg, modality_loss[target_neg])

        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)
        num_pos = target.sum(dim=1)
        num_neg = logits.size(0) - num_pos

        return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()


def _get_dtype_max(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return torch.finfo(tensor.dtype).max
    if not tensor.is_complex():
        return torch.iinfo(tensor.dtype).max
    raise ValueError(
        f"Unsupported dtype {tensor.dtype}. Only floating point and integer types are supported."
    )


def _is_all_dtype_max(tensor: torch.Tensor) -> torch.BoolTensor:
    dtype_max = _get_dtype_max(tensor)
    return torch.all(tensor == dtype_max)


def _gather_dicts(
    dicts: dict[str, torch.Tensor],
    local_loss: bool,
    rank: int,
    gather_with_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Gather dictionaries of tensors across all processes.

    Parameters
    ----------
    dicts : dict[str, torch.Tensor]
        Dictionary of tensors to gather.
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
    dict[str, torch.Tensor]
        Gathered dictionary of tensors.
    """
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    current_device = next(iter(dicts.values())).device
    dist.barrier(group=group)

    # gather keys
    local_keys = list(dicts.keys())
    all_keys: list[str] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(all_keys, local_keys, group=group)
    all_keys = sorted(set(itertools.chain.from_iterable(all_keys)))

    # gather tensors
    gathered_dict: dict[str, torch.Tensor] = {}
    for key in all_keys:
        if key not in dicts:  # use dummy tensor for missing key in current process
            placeholder_tensor = dicts[local_keys[0]]
            tensor = torch.full_like(
                placeholder_tensor,
                fill_value=_get_dtype_max(placeholder_tensor),
                device=current_device,
                memory_format=torch.contiguous_format,
                requires_grad=gather_with_grad
                and placeholder_tensor.is_floating_point(),  # only floating point tensors can have gradients
            )
        else:
            tensor = dicts[key].contiguous()

        gathered_tensors: list[torch.Tensor] = _gather_all_tensors(
            tensor,
            world_size=world_size,
            group=group,
            gather_with_grad=gather_with_grad,
        )

        if not gather_with_grad and not local_loss:
            gathered_tensors[rank] = tensor

        # filter out placeholder tensors
        gathered_tensors = [t for t in gathered_tensors if not _is_all_dtype_max(t)]

        gathered_dict[key] = torch.cat(gathered_tensors, dim=0)

    return gathered_dict


def _simple_gather_all_tensors(
    result: torch.Tensor, group: Any, world_size: int, gather_with_grad: bool = False
) -> list[torch.Tensor]:
    if gather_with_grad:
        return list(dist_nn.all_gather(result, group))

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result


def _gather_all_tensors(
    a_tensor: torch.Tensor,
    world_size: Optional[int] = None,
    group: Optional[Any] = None,
    gather_with_grad: bool = False,
) -> list[torch.Tensor]:
    """Gather tensor(s) from all devices onto a list and broadcast to all devices.

    Parameters
    ----------
    a_tensor : torch.Tensor
        The tensor to gather.
    world_size : int, default=None
        Number of processes in the group.
    group : Any, default=None
        The process group to work on.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.

    Returns
    -------
    list[torch.Tensor]
        List of gathered tensors.
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    a_tensor = a_tensor.contiguous()

    if world_size is None:
        world_size = dist.get_world_size(group)
        dist.barrier(group=group)

    # if the tensor is scalar, things are easy
    if a_tensor.ndim == 0:
        return _simple_gather_all_tensors(a_tensor, group, world_size, gather_with_grad)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(a_tensor.shape, device=a_tensor.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(a_tensor, group, world_size, gather_with_grad)

    # 3. If not, we need to pad each local tensor to maximum size, gather and
    # then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(a_tensor, pad_dims)
    if gather_with_grad:
        gathered_result = list(dist_nn.all_gather(result_padded, group))
    else:
        gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
        dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result
