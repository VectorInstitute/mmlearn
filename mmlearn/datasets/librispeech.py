"""LibriSpeech dataset."""

import os

import torch
import torch.nn.functional as F  # noqa: N812
from hydra_zen import MISSING, store
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data.dataset import Dataset

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


_TORCHAUDIO_AVAILABLE = RequirementCache("torchaudio>=2.4.0")
SAMPLE_RATE = 16000


def pad_or_trim(
    array: torch.Tensor,
    length: int = 30 * SAMPLE_RATE,
    *,
    axis: int = -1,
) -> torch.Tensor:
    """Pad or trim the audio array to `length` along the given axis.

    Adapted from: https://github.com/openai/whisper/blob/main/whisper/audio.py#L65C1-L88C17

    Parameters
    ----------
    array : torch.Tensor
        Audio array.
    length : int, default=480000
        Length to pad or trim to. Defaults to 30 seconds at 16 kHz.
    axis : int, default=-1
        Axis along which to pad or trim.

    Returns
    -------
    array : torch.Tensor
        Padded or trimmed audio array.
    """
    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis,
            index=torch.arange(length, device=array.device),
        )

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    return array


@store(
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("LIBRISPEECH_ROOT_DIR", MISSING),
)
class LibriSpeech(Dataset[Example]):
    """LibriSpeech dataset.

    This is a wrapper around `torchaudio.datasets.LIBRISPEECH` that assumes that
    the dataset is already downloaded and the top-level directory of the dataset
    in the root directory is `librispeech`.
    This class only returns the audio and transcript from the dataset.

    Parameters
    ----------
    root_dir : str
        Root directory of dataset.
    split : {"train-clean-100", "train-clean-360", "train-other-500", "dev-clean",
        "dev-other", "test-clean", "test-other"}, default="train-clean-100"
        Split of the dataset to use.

    """

    def __init__(self, root_dir: str, split: str = "train-clean-100") -> None:
        """Initialize LibriSpeech dataset."""
        super().__init__()
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "LibriSpeech dataset requires `torchaudio` which is not installed."
            )
        from torchaudio.datasets import LIBRISPEECH

        self.dataset = LIBRISPEECH(
            root=root_dir,
            url=split,
            download=False,
            folder_in_archive="librispeech",
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Example:
        """Return an example from the dataset."""
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        assert (
            sample_rate == SAMPLE_RATE
        ), f"Expected sample rate to be `16000`, got {sample_rate}."
        waveform = pad_or_trim(waveform.flatten())

        return Example(
            {
                Modalities.AUDIO: waveform,
                Modalities.TEXT: transcript,
                EXAMPLE_INDEX_KEY: idx,
            },
        )
