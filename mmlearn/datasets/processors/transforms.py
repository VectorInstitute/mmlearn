"""Custom transforms for datasets."""

import math
from typing import List, Union

import torch
from hydra_zen import store


@store(group="datasets/transforms", provider="mmlearn")
class TrimText:
    """Trim text strings as a preprocessing step before tokenization."""

    def __init__(self, trim_size: int) -> None:
        """Initialize the object."""
        self.trim_size = trim_size

    def __call__(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
        """Trim the given sentence(s)."""
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


def _no_grad_trunc_normal_(
    tensor: torch.Tensor, mean: float, std: float, a: float, b: float
) -> torch.Tensor:
    """
    Apply truncated normal initialization to a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be initialized.
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
    a : float
        Minimum value of the truncated distribution.
    b : float
        Maximum value of the truncated distribution.

    Returns
    -------
    torch.Tensor
        The initialized tensor.
    """

    def norm_cdf(x: float) -> float:
        """Compute standard normal cumulative distribution function."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower_limit = norm_cdf((a - mean) / std)
        upper_limit = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower_limit - 1, 2 * upper_limit - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """
    Initialize a tensor using a truncated normal distribution.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be initialized.
    mean : float, default=0.
        Mean of the normal distribution.
    std : float, default=1.
        Standard deviation of the normal distribution.
    a : float, default=-2.
        Minimum value of the truncated distribution.
    b : float, default=2.
        Maximum value of the truncated distribution.

    Returns
    -------
    torch.Tensor
        The initialized tensor.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x: torch.Tensor, b: int, repeat: int) -> torch.Tensor:
    """
    Repeat and interleave a tensor across the batch dimension.

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
