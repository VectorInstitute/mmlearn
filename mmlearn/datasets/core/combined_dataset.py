"""Wrapper for combining multiple datasets into one."""

import bisect
from collections.abc import Iterator
from typing import Iterable, Union

import numpy as np
from torch.utils._pytree import tree_flatten
from torch.utils.data import Dataset, IterableDataset

from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core.dataset_info import DatasetInfo


class CombinedDataset(Dataset[Example]):
    """Combine multiple datasets into one.

    This class is similar to `torch.utils.data.ConcatDataset` but allows for
    combining iterable-style datasets with map-style datasets. The iterable-style
    datasets must implement the `__len__` method, which is used to determine the
    total length of the combined dataset. When an index is passed to the combined
    dataset, the dataset that contains the example at that index is determined and
    the example is retrieved from that dataset. Since iterable-style datasets do
    not support random access, the examples are retrieved sequentially from the
    iterable-style datasets. When the end of an iterable-style dataset is reached,
    the iterator is reset and the next example is retrieved from the beginning of
    the dataset.


    Parameters
    ----------
    datasets : Iterable[Union[Dataset, IterableDataset]]
        Iterable of datasets to combine.

    """

    def __init__(
        self, datasets: Iterable[Union[Dataset[Example], IterableDataset[Example]]]
    ) -> None:
        """Initialize the combined dataset."""
        self.datasets, _ = tree_flatten(datasets)
        if not all(
            isinstance(dataset, (Dataset, IterableDataset)) for dataset in self.datasets
        ):
            raise TypeError(
                "Expected argument `datasets` to be an iterable of `Dataset` or "
                f"`IterableDataset` instances, but found: {self.datasets}",
            )
        if len(self.datasets) == 0:
            raise ValueError(
                "Expected a non-empty iterable of datasets but found an empty iterable",
            )

        self.cumulative_sizes: list[int] = np.cumsum(
            [len(dataset) for dataset in self.datasets]
        ).tolist()
        self._iterators: list[Iterator[Example]] = []
        self._iter_dataset_mapping: dict[int, int] = {}

        # create iterators for iterable datasets and map dataset index to iterator index
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, IterableDataset):
                self._iterators.append(iter(dataset))
                self._iter_dataset_mapping[idx] = len(self._iterators) - 1

    def __getitem__(self, idx: int) -> Example:
        """Return an example from the combined dataset."""
        if idx < 0:  # handle negative indices
            if -idx > len(self):
                raise IndexError(
                    f"Index {idx} is out of bounds for the combined dataset with "
                    f"length {len(self)}",
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        curr_dataset = self.datasets[dataset_idx]
        if isinstance(curr_dataset, IterableDataset):
            iter_idx = self._iter_dataset_mapping[dataset_idx]
            try:
                example = next(self._iterators[iter_idx])
            except StopIteration:
                self._iterators[iter_idx] = iter(curr_dataset)
                example = next(self._iterators[iter_idx])
        else:
            if dataset_idx == 0:
                example_idx = idx
            else:
                example_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            example = curr_dataset[example_idx]

        if not isinstance(example, Example):
            raise TypeError(
                "Expected dataset examples to be instances of `Example` "
                f"but found {type(example)}",
            )

        if not hasattr(example, "dataset_index"):
            example.dataset_index = dataset_idx
        if not hasattr(example, "example_ids"):
            example.create_ids()

        return example

    def __len__(self) -> int:
        """Return the total number of examples in the combined dataset."""
        return self.cumulative_sizes[-1]
    
    def create_all_dataset_info(self):
        all_dataset_info = {}
        for _, dataset in enumerate(self.datasets):
            label_mapping = dataset.get_label_mapping()
            dataset_info = DatasetInfo(class_count=len(label_mapping), label_mapping=label_mapping)
            all_dataset_info[dataset.name()] = dataset_info

        return all_dataset_info
