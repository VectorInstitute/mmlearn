"""Tests for the combined dataset object and sampler."""

from collections.abc import Iterator

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from mmlearn.datasets.core import (
    CombinedDataset,
    CombinedDatasetRatioSampler,
    DefaultDataCollator,
    Example,
)


class DummyIterableDataset(IterableDataset):
    """Dummy iterable dataset."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        super().__init__()
        self.examples = [10, 20, 30, 40, 50, 60, 70]

    def __iter__(self) -> Iterator[Example]:
        """Yield dummy examples."""
        for example in self.examples:
            yield Example({"tens": example})

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.examples)


class DummyNegativesDataset(Dataset):
    """Dummy dataset with negative examples."""

    def __init__(self) -> None:
        super().__init__()
        self.tensors = torch.tensor([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])

    def __getitem__(self, index: int) -> Example:
        """Return an example from the dataset."""
        return Example({"negs": self.tensors[index]})

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tensors)


def test_combined_dataset():
    """Test the combined dataset object."""
    dataset1 = DummyNegativesDataset()
    dataset2 = DummyIterableDataset()
    combined_dataset = CombinedDataset([dataset1, dataset2])
    assert len(combined_dataset.datasets) == 2

    with pytest.warns(
        UserWarning,
        match="Cannot create `example_ids` without `example_index` and `dataset_index`.*",
    ):
        example = combined_dataset[0]
    assert isinstance(example, Example)
    assert hasattr(example, "negs")
    assert torch.equal(example.negs, torch.tensor(-1))
    assert example.dataset_index == 0

    with pytest.warns(
        UserWarning,
        match="Cannot create `example_ids` without `example_index` and `dataset_index`.*",
    ):
        example = combined_dataset[-1]
    assert isinstance(example, Example)
    assert hasattr(example, "tens")
    assert example.tens == 10
    assert example.dataset_index == 1

    non_dataset = "not a dataset"
    with pytest.raises(TypeError):
        combined_dataset = CombinedDataset([dataset1, dataset2, non_dataset])

    with pytest.raises(ValueError):
        combined_dataset = CombinedDataset([])


@pytest.mark.integration_test()
class TestCombinedDatasetRatioSampler:
    """Test the combined dataset ratio sampler."""

    def test_combined_dataset_ratio_sampler(self):
        """Test the combined dataset ratio sampler."""
        dataset1 = DummyNegativesDataset()
        dataset2 = DummyIterableDataset()
        combined_dataset = CombinedDataset([dataset1, dataset2])
        sampler = CombinedDatasetRatioSampler(
            combined_dataset, [0.5, 0.5], num_samples=10, rank=0, num_replicas=1
        )

        assert len(sampler) == 10
        iterator = iter(sampler)

        # get 10 samples
        sample_indices = [next(iterator) for _ in range(10)]

        # half the indices should be for the negatives and half should be positives
        num_negs = sum(1 for index in sample_indices if index < 10)
        num_pos = sum(1 for index in sample_indices if index >= 10)
        assert num_negs == 5
        assert num_pos == 5

    def test_combined_dataset_ratio_sampler_single_batch(self):
        """Test the combined dataset ratio sampler with a single batch."""
        dataset1 = DummyNegativesDataset()
        dataset2 = DummyIterableDataset()
        combined_dataset = CombinedDataset([dataset1, dataset2])
        # oversample the iterable dataset to test StopIteration handling
        sampler = CombinedDatasetRatioSampler(
            combined_dataset,
            [0.5, 0.5],
            shuffle=False,  # dataset samples will appear in order
            num_samples=14,  # 7 from each dataset
            rank=0,
            num_replicas=1,
        )
        dataloader = DataLoader(
            combined_dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=DefaultDataCollator(),
        )

        with pytest.warns(
            UserWarning,
            match="Cannot create `example_ids` without `example_index` and `dataset_index`.*",
        ):
            for i, batch in enumerate(dataloader):
                if i < 7:
                    assert "negs" in batch
                    assert torch.all(batch["negs"] < 0)
                else:
                    assert "tens" in batch
                    assert torch.all(batch["tens"] % 10 == 0)

    def test_combined_dataset_ratio_sampler_batched_multiple_epochs(self):
        """Test the combined dataset ratio sampler with multiple epochs."""
        dataset1 = DummyNegativesDataset()
        dataset2 = DummyIterableDataset()
        combined_dataset = CombinedDataset([dataset1, dataset2])
        # oversample both datasets
        sampler = CombinedDatasetRatioSampler(
            combined_dataset,
            [0.5, 0.5],
            shuffle=False,
            num_samples=34,  # 17 from each dataset
            rank=0,
            num_replicas=1,
            replacement=True,
        )
        dataloader = DataLoader(
            combined_dataset,
            batch_size=2,
            sampler=sampler,
            collate_fn=DefaultDataCollator(),
        )

        with pytest.warns(
            UserWarning,
            match="Cannot create `example_ids` without `example_index` and `dataset_index`.*",
        ):
            for _ in range(2):
                for i, batch in enumerate(dataloader):
                    if i < 9:
                        assert "negs" in batch
                        assert torch.all(batch["negs"] < 0)
                    else:
                        assert "tens" in batch
                        assert torch.all(batch["tens"] % 10 == 0)
