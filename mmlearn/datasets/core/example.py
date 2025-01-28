"""Module for example-related classes and functions."""

from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Hashable, Optional

import torch
from lightning.fabric.utilities import rank_zero_warn


class Example(OrderedDict[Any, Any]):
    """A representation of a single example from a dataset.

    This class is a subclass of :py:class:`~collections.OrderedDict` and provides
    attribute-style access. This means that `example["text"]` and `example.text`
    are equivalent. All datasets in this library return examples as
    :py:class:`~mmlearn.datasets.core.example.Example` objects.


    Parameters
    ----------
    init_dict : Optional[MutableMapping[Hashable, Any]], optional, default=None
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
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def create_ids(self) -> None:
        """Create a unique id for the example from the dataset and example index.

        This method combines the dataset index and example index to create an
        attribute called `example_ids`, which is a dictionary of tensors. The
        dictionary keys are all the keys in the example except for `example_ids`,
        `example_index`, and `dataset_index`. The values are tensors of shape `(2,)`
        containing the tuple `(dataset_index, example_index)` for each key.
        The `example_ids` is used to (re-)identify pairs of examples from different
        modalities after they have been combined into a batch.

        Warns
        -----
        UserWarning
            If the `example_index` and `dataset_index` attributes are not set.

        Notes
        -----
        - The Example must have the following attributes set before calling this
          this method: `example_index` (usually set/returned by the dataset) and
          `dataset_index` (usually set by the :py:class:`~mmlearn.datasets.core.combined_dataset.CombinedDataset` object)
        - The :py:func:`~mmlearn.datasets.core.example.find_matching_indices`
          function can be used to find matching examples given two tensors of example ids.

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


def find_matching_indices(
    first_example_ids: torch.Tensor, second_example_ids: torch.Tensor
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
