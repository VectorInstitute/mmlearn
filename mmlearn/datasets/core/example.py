"""Module for example-related classes and functions."""

from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Hashable, Optional

import torch
from lightning.fabric.utilities import rank_zero_warn
from torch.utils.data import default_collate


class Example(OrderedDict[Any, Any]):
    """A representation of a single example from a dataset.

    This class is a subclass of `OrderedDict` and provides attribute-style access.
    This means that `example["text"]` and `example.text` are equivalent.
    All datasets in this library return examples as `Example` objects.


    Parameters
    ----------
    init_dict : Mapping[str, Any], optional, default=None
        Dictionary to init `Example` class with.

    Examples
    --------
    >>> example = Example({"text": torch.tensor(2)})
    >>> example.text.zero_()
    tensor(0)
    >>> example.context = torch.tensor(4)  # set custom attributes after initialization
    """

    def __init__(
        self,
        init_dict: Optional[MutableMapping[Hashable, Any]] = None,
    ) -> None:
        """Initialize a `Example` object."""
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def create_ids(self) -> None:
        """Create a unique id for the example from the dataset and example index.

        The example id is a tensor of shape `(2,)` (or a tuple) that contains the
        dataset index and example index. It can be used to (re-)identify pairs of
        examples from different modalities.
        The example must have the `example_index` (usually set/returned by the dataset)
        and `dataset_index` (usually set by the `dataset.combined_dataset.CombinedDataset`
        object) attributes set before calling this method.
        The matching example ids can be found using the `dataset.example.find_matching_indices`
        method.

        Warns
        -----
        UserWarning
            If the `example_index` and `dataset_index` attributes are not set.
        """  # noqa: W505
        if hasattr(self, "example_index") and hasattr(self, "dataset_index"):
            self.example_ids = {
                key: torch.tensor([self.dataset_index, self.example_index])
                for key in self.keys()
                if key not in ("example_ids", "example_index", "dataset_index")
            }
        else:
            rank_zero_warn(
                "Cannot create `example_ids` without `example_index` and `dataset_index` "
                "attributes. Set these attributes before calling `create_ids`. "
                "No `example_ids` was created.",
                stacklevel=2,
                category=UserWarning,
            )

    def __getattr__(self, key: str) -> Any:
        """Get attribute by key."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        """Set attribute by key."""
        if isinstance(value, MutableMapping):
            value = Example(value)
        self[key] = value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Set item by key."""
        if isinstance(value, MutableMapping):
            value = Example(value)
        super().__setitem__(key, value)


def _merge_examples(examples: list[Example]) -> dict[str, Any]:
    """Convert a list of `dataset.example.Example` objects into a dictionary.

    This method will merge examples with the same key into a list.

    Parameters
    ----------
    examples : list[Example]
        List of examples to convert.

    Returns
    -------
    dict[str, Any]
        Dictionary of converted examples.
    """
    merged_examples: dict[str, Any] = {}
    for example in examples:
        for key in example:
            if key in merged_examples:
                merged_examples[key].append(example[key])
            else:
                merged_examples[key] = [example[key]]

    for key in merged_examples:
        if isinstance(merged_examples[key][0], Example):
            merged_examples[key] = _merge_examples(merged_examples[key])

    return merged_examples


def _collate_example_dict(examples: dict[str, Any]) -> dict[str, Any]:
    """Collate a dictionary of examples into a batch.

    Parameters
    ----------
    examples : dict[str, Any]
        Dictionary of examples to collate.

    Returns
    -------
    dict[str, Any]
        Dictionary of collated examples.
    """
    batch = {}
    for k, v in examples.items():
        if isinstance(v, dict):
            batch[k] = _collate_example_dict(v)
        else:
            batch[k] = default_collate(v)

    return batch


def collate_example_list(examples: list[Example]) -> dict[str, Any]:
    """Collate a list of `Example` objects into a batch.

    Parameters
    ----------
    examples : list[Example]
        List of examples to collate.

    Returns
    -------
    dict[str, Any]
        Dictionary of batched examples.

    """
    return _collate_example_dict(_merge_examples(examples))


def find_matching_indices(
    first_example_ids: torch.Tensor,
    second_example_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the indices of matching examples given two tensors of example ids.

    Matching examples are defined as examples with the same value in both tensors.
    This method is useful for finding pairs of examples from different modalities
    that are related to each other in a batch.

    Parameters
    ----------
    first_example_ids : torch.Tensor
        A tensor of example ids of shape `(N, 2)`, where `N` is the number of examples.
    second_example_ids : torch.Tensor
        A tensor of example ids of shape `(M, 2)`, where `M` is the number of examples.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of tensors containing the indices of matching examples in the first and
        second tensor, respectively.

    Raises
    ------
    TypeError
        If either `first_example_ids` or `second_example_ids` is not a tensor.
    ValueError
        If either `first_example_ids` or `second_example_ids` is not a 2D tensor
        with the second dimension having a size of `2`.

    Examples
    --------
    >>> img_example_ids = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)])
    >>> text_example_ids = torch.tensor([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    >>> find_matching_indices(img_example_ids, text_example_ids)
    (tensor([2, 3]), tensor([0, 1]))


    """
    if not isinstance(first_example_ids, torch.Tensor) or not isinstance(
        second_example_ids,
        torch.Tensor,
    ):
        raise TypeError(
            f"Expected inputs to be tensors, but got {type(first_example_ids)} "
            f"and {type(second_example_ids)}.",
        )
    val = 2
    if not (first_example_ids.ndim == val and first_example_ids.shape[1] == val):
        raise ValueError(
            "Expected argument `first_example_ids` to be a tensor of shape (N, 2), "
            f"but got shape {first_example_ids.shape}.",
        )
    if not (second_example_ids.ndim == val and second_example_ids.shape[1] == val):
        raise ValueError(
            "Expected argument `second_example_ids` to be a tensor of shape (N, 2), "
            f"but got shape {second_example_ids.shape}.",
        )

    first_example_ids = first_example_ids.unsqueeze(1)  # shape=(N, 1, 2)
    second_example_ids = second_example_ids.unsqueeze(0)  # shape=(1, M, 2)

    # compare all elements; results in a shape (N, M) tensor
    matches = torch.all(first_example_ids == second_example_ids, dim=-1)
    first_indices, second_indices = torch.where(matches)
    return first_indices, second_indices
