"""Custom transforms for datasets/inputs."""

from typing import Union

import torch
from hydra_zen import store


@store(group="datasets/transforms", provider="mmlearn")
class TrimText:
    """Trim text strings as a preprocessing step before tokenization.

    Parameters
    ----------
    trim_size : int
        The maximum length of the trimmed text.
    """

    def __init__(self, trim_size: int) -> None:
        self.trim_size = trim_size

    def __call__(self, sentence: Union[str, list[str]]) -> Union[str, list[str]]:
        """Trim the given sentence(s).

        Parameters
        ----------
        sentence : Union[str, list[str]]
            Sentence(s) to be trimmed.

        Returns
        -------
        Union[str, list[str]]
            Trimmed sentence(s).

        Raises
        ------
        TypeError
            If the input sentence is not a string or list of strings.
        """
        if not isinstance(sentence, (list, str)):
            raise TypeError(
                "Expected argument `sentence` to be a string or list of strings, "
                f"but got {type(sentence)}"
            )

        if isinstance(sentence, str):
            return sentence[: self.trim_size]

        for i, s in enumerate(sentence):
            sentence[i] = s[: self.trim_size]

        return sentence


def repeat_interleave_batch(x: torch.Tensor, b: int, repeat: int) -> torch.Tensor:
    """Repeat and interleave a tensor across the batch dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be repeated.
    b : int
        Size of the batch to be repeated.
    repeat : int
        Number of times to repeat each batch.

    Returns
    -------
    torch.Tensor
        The repeated tensor with shape adjusted for the batch.
    """
    n = len(x) // b
    return torch.cat(
        [
            torch.cat([x[i * b : (i + 1) * b] for _ in range(repeat)], dim=0)
            for i in range(n)
        ],
        dim=0,
    )
