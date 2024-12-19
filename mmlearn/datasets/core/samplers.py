"""Samplers for data loading."""

import math
from typing import Iterator, Optional, Sequence, Sized

import torch
import torch.distributed as dist
from hydra_zen import store
from torch.utils.data import Dataset, Sampler

from mmlearn.datasets.core.combined_dataset import CombinedDataset


@store(group="dataloader/sampler", provider="mmlearn")
class CombinedDatasetRatioSampler(Sampler[int]):
    """Sampler for weighted sampling from a :py:class:`~mmlearn.datasets.core.combined_dataset.CombinedDataset`.

    Parameters
    ----------
    dataset : CombinedDataset
        An instance of :py:class:`~mmlearn.datasets.core.combined_dataset.CombinedDataset`
        to sample from.
    ratios : Optional[Sequence[float]], optional, default=None
        A sequence of ratios for sampling from each dataset in the combined dataset.
        The length of the sequence must be equal to the number of datasets in the
        combined dataset (`dataset`). If `None`, the length of each dataset in the
        combined dataset is used as the ratio. The ratios are normalized to sum to 1.
    num_samples : Optional[int], optional, default=None
        The number of samples to draw from the combined dataset. If `None`, the
        sampler will draw as many samples as there are in the combined dataset.
        This number must yield at least one sample per dataset in the combined
        dataset, when multiplied by the corresponding ratio.
    replacement : bool, default=False
        Whether to sample with replacement or not.
    shuffle : bool, default=True
        Whether to shuffle the sampled indices or not. If `False`, the indices of
        each dataset will appear in the order they are stored in the combined dataset.
        This is similar to sequential sampling from each dataset. The datasets
        that make up the combined dataset are still sampled randomly.
    rank : Optional[int], optional, default=None
        Rank of the current process within :attr:`num_replicas`. By default,
        :attr:`rank` is retrieved from the current distributed group.
    num_replicas : Optional[int], optional, default=None
        Number of processes participating in distributed training. By
        default, :attr:`num_replicas` is retrieved from the current distributed group.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch or not. If `True`, the sampler will
        drop samples to make the number of samples evenly divisible by the number of
        replicas in distributed mode.
    seed : int, default=0
        Random seed used to when sampling from the combined dataset and shuffling
        the sampled indices.

    Attributes
    ----------
    dataset : CombinedDataset
        The dataset to sample from.
    num_samples : int
        The number of samples to draw from the combined dataset.
    probs : torch.Tensor
        The probabilities for sampling from each dataset in the combined dataset.
        This is computed from the `ratios` argument and is normalized to sum to 1.
    replacement : bool
        Whether to sample with replacement or not.
    shuffle : bool
        Whether to shuffle the sampled indices or not.
    rank : int
        Rank of the current process within :attr:`num_replicas`.
    num_replicas : int
        Number of processes participating in distributed training.
    drop_last : bool
        Whether to drop samples to make the number of samples evenly divisible by the
        number of replicas in distributed mode.
    seed : int
        Random seed used to when sampling from the combined dataset and shuffling
        the sampled indices.
    epoch : int
        Current epoch number. This is used to set the random seed. This is useful
        in distributed mode to ensure that each process receives a different random
        ordering of the samples.
    total_size : int
        The total number of samples across all processes.
    """  # noqa: W505

    def __init__(  # noqa: PLR0912
        self,
        dataset: CombinedDataset,
        ratios: Optional[Sequence[float]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = False,
        shuffle: bool = True,
        rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if not isinstance(dataset, CombinedDataset):
            raise TypeError(
                "Expected argument `dataset` to be of type `CombinedDataset`, "
                f"but got {type(dataset)}.",
            )
        if not isinstance(seed, int):
            raise TypeError(
                f"Expected argument `seed` to be an integer, but got {type(seed)}.",
            )
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self._num_samples = num_samples
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "Expected argument `num_samples` to be a positive integer, but got "
                f"{self.num_samples}.",
            )

        if ratios is None:
            ratios = [len(subset) for subset in self.dataset.datasets]

        num_datasets = len(self.dataset.datasets)
        if len(ratios) != num_datasets:
            raise ValueError(
                f"Expected argument `ratios` to be of length {num_datasets}, "
                f"but got length {len(ratios)}.",
            )
        prob_sum = sum(ratios)
        if not all(ratio >= 0 for ratio in ratios) and prob_sum > 0:
            raise ValueError(
                "Expected argument `ratios` to be a sequence of non-negative numbers. "
                f"Got {ratios}.",
            )
        self.probs = torch.tensor(
            [ratio / prob_sum for ratio in ratios],
            dtype=torch.double,
        )
        if any((prob * self.num_samples) <= 0 for prob in self.probs):
            raise ValueError(
                "Expected dataset ratio to result in at least one sample per dataset. "
                f"Got dataset sizes {self.probs * self.num_samples}.",
            )

    @property
    def num_samples(self) -> int:
        """Return the number of samples managed by the sampler."""
        # dataset size might change at runtime
        if self._num_samples is None:
            num_samples = len(self.dataset)
        else:
            num_samples = self._num_samples

        if self.drop_last and num_samples % self.num_replicas != 0:
            # split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            num_samples = math.ceil(
                (num_samples - self.num_replicas) / self.num_replicas,
            )
        else:
            num_samples = math.ceil(num_samples / self.num_replicas)
        return num_samples

    @property
    def total_size(self) -> int:
        """Return the total size of the dataset."""
        return self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Return an iterator that yields sample indices for the combined dataset."""
        generator = torch.Generator()
        seed = self.seed + self.epoch
        generator.manual_seed(seed)

        cumulative_sizes = [0] + self.dataset._cumulative_sizes
        num_samples_per_dataset = [int(prob * self.total_size) for prob in self.probs]
        indices = []
        for i in range(len(self.dataset.datasets)):
            per_dataset_indices: torch.Tensor = torch.multinomial(
                torch.ones(cumulative_sizes[i + 1] - cumulative_sizes[i]),
                num_samples_per_dataset[i],
                replacement=self.replacement,
                generator=generator,
            )
            # adjust indices to reflect position in cumulative dataset
            per_dataset_indices += cumulative_sizes[i]
            assert per_dataset_indices.max() < cumulative_sizes[i + 1], (
                f"Indices from dataset {i} exceed dataset size. "
                f"Got indices {per_dataset_indices} and dataset size {cumulative_sizes[i + 1]}.",
            )
            indices.append(per_dataset_indices)

        indices = torch.cat(indices)
        if self.shuffle:
            rand_indices = torch.randperm(len(indices), generator=generator)
            indices = indices[rand_indices]

        indices = indices.tolist()  # type: ignore[attr-defined]
        num_indices = len(indices)

        if num_indices < self.total_size:
            padding_size = self.total_size - num_indices
            if padding_size <= num_indices:
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / num_indices))[
                    :padding_size
                ]
        elif num_indices > self.total_size:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples, (
            f"Expected {self.num_samples} samples, but got {len(indices)}.",
        )

        yield from iter(indices)

    def __len__(self) -> int:
        """Return the total number of samples in the sampler."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random
        ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering.

        Parameters
        ----------
        epoch : int
            Epoch number.

        """
        self.epoch = epoch

        # some iterable datasets (especially huggingface iterable datasets) might
        # require setting the epoch to ensure shuffling works properly
        for dataset in self.dataset.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


@store(group="dataloader/sampler", provider="mmlearn")
class DistributedEvalSampler(Sampler[int]):
    """Sampler for distributed evaluation.

    The main differences between this and :py:class:`torch.utils.data.DistributedSampler`
    are that this sampler does not add extra samples to make it evenly divisible and
    shuffling is disabled by default.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset used for sampling.
    num_replicas : Optional[int], optional, default=None
        Number of processes participating in distributed training. By
        default, :attr:`rank` is retrieved from the current distributed group.
    rank : Optional[int], optional, default=None
        Rank of the current process within :attr:`num_replicas`. By default,
        :attr:`rank` is retrieved from the current distributed group.
    shuffle : bool, optional, default=False
        If `True` (default), sampler will shuffle the indices.
    seed : int, optional, default=0
        Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        This number should be identical across all processes in the
        distributed group.

    Warnings
    --------
    DistributedEvalSampler should NOT be used for training. The distributed processes
    could hang forever. See [1]_ for details

    Notes
    -----
    - This sampler is for evaluation purpose where synchronization does not happen
      every epoch. Synchronization should be done outside the dataloader loop.
      It is especially useful in conjunction with
      :py:class:`torch.nn.parallel.DistributedDataParallel` [2]_.
    - The input Dataset is assumed to be of constant size.
    - This implementation is adapted from [3]_.

    References
    ----------
    .. [1] https://github.com/pytorch/pytorch/issues/22584
    .. [2] https://discuss.pytorch.org/t/how-to-validate-in-distributeddataparallel-correctly/94267/11
    .. [3] https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py


    Examples
    --------
    >>> def example():
    ...     start_epoch, n_epochs = 0, 2
    ...     sampler = DistributedEvalSampler(dataset) if is_distributed else None
    ...     loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    ...     for epoch in range(start_epoch, n_epochs):
    ...         if is_distributed:
    ...             sampler.set_epoch(epoch)
    ...         evaluate(loader)

    """  # noqa: W505

    def __init__(
        self,
        dataset: Dataset[Sized],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

    @property
    def total_size(self) -> int:
        """Return the total size of the dataset."""
        return len(self.dataset)

    @property
    def num_samples(self) -> int:
        """Return the number of samples managed by the sampler."""
        indices = list(range(self.total_size))[
            self.rank : self.total_size : self.num_replicas
        ]
        return len(indices)  # true value without extra samples

    def __iter__(self) -> Iterator[int]:
        """Return an iterator that iterates over the indices of the dataset."""
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random
        ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering.

        Parameters
        ----------
        epoch : int
            Epoch number.

        """
        self.epoch = epoch
