"""Data collators for batching examples."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch.utils.data import default_collate

from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core.modalities import Modalities


@dataclass
class DefaultDataCollator:
    """Default data collator for batching examples.

    This data collator will collate a list of :py:class:`~mmlearn.datasets.core.example.Example`
    objects into a batch. It can also apply processing functions to specified keys
    in the batch before returning it.

    Parameters
    ----------
    batch_processors : Optional[dict[str, Callable[[Any], Any]]], optional, default=None
        Dictionary of callables to apply to the batch before returning it.

    Raises
    ------
    ValueError
        If the batch processor for a key does not return a dictionary with the
        key in it.
    """  # noqa: W505

    #: Dictionary of callables to apply to the batch before returning it.
    #: The key is the name of the key in the batch, and the value is the processing
    #: function to apply to the key. The processing function must take a single
    #: argument and return a single value. If the processing function returns
    #: a dictionary, it must contain the key that was processed in it (all the
    #: other keys will also be included in the batch).
    batch_processors: Optional[dict[str, Callable[[Any], Any]]] = None

    def __call__(self, examples: list[Example]) -> dict[str, Any]:
        """Collate a list of `Example` objects and apply processing functions."""
        batch = collate_example_list(examples)

        if self.batch_processors is not None:
            for key, processor in self.batch_processors.items():
                batch_key: str = key
                if Modalities.has_modality(key):
                    batch_key = Modalities.get_modality(key).name

                if batch_key in batch:
                    batch_processed = processor(batch[batch_key])
                    if isinstance(batch_processed, Mapping):
                        if batch_key not in batch_processed:
                            raise ValueError(
                                f"Batch processor for '{key}' key must return a dictionary "
                                f"with '{batch_key}' in it."
                            )
                        batch.update(batch_processed)
                    else:
                        batch[batch_key] = batch_processed

        return batch


def collate_example_list(examples: list[Example]) -> dict[str, Any]:
    """Collate a list of :py:class:`~mmlearn.datasets.core.example.Example` objects into a batch.

    Parameters
    ----------
    examples : list[Example]
        list of examples to collate.

    Returns
    -------
    dict[str, Any]
        Dictionary of batched examples.

    """  # noqa: W505
    return _collate_example_dict(_merge_examples(examples))


def _merge_examples(examples: list[Example]) -> dict[str, Any]:
    """Convert a list of `dataset.example.Example` objects into a dictionary.

    This method will merge examples with the same key into a list.

    Parameters
    ----------
    examples : list[Example]
        list of examples to convert.

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
