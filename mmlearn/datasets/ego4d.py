"""Ego4D dataset."""

import glob
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from hydra_zen import MISSING, store
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torchvision import transforms

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@store(
    name="Ego4D",
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("EGO4D_ROOT_DIR", MISSING),
)
class Ego4DDataset(Dataset[Example]):
    """A PyTorch Dataset for loading and processing videos from the Ego4D dataset.

    Parameters
    ----------
    root_dir : List[str]
        Path to the root directory containing the video files.
    clip_duration : int
        Duration of each video clip in seconds.
    clips_per_video : int
        Number of clips to sample from each video.
    sample_rate : int
        Sample rate for audio processing.
    video_transform : Optional[Callable], default=None
        A callable that takes in a video clip and returns a transformed version of it.
    """

    def __init__(
        self,
        root_dir: str,
        clip_duration: int = 2,
        clips_per_video: int = 5,
        sample_rate: int = 16000,
        video_transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the dataset."""
        self.video_paths = glob.glob(os.path.join(root_dir, "*.mp4"))
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.sample_rate = sample_rate

        if video_transform is not None:
            self.video_transform = video_transform
        else:
            self.video_transform = transforms.Compose(
                [
                    pv_transforms.ShortSideScale(224),
                    pv_transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ],
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Example:
        """Return a video clip from the dataset."""
        video_path = self.video_paths[idx]
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)

        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=self.clip_duration,
            clips_per_video=self.clips_per_video,
        )
        all_clips_timepoints = self._get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for start, end in all_clips_timepoints:
            clip = video.get_clip(start, end)
            if clip is None:
                raise ValueError("No clip found")
            video_clip = clip["video"] / 255.0  # Normalizing
            video_clip = self.video_transform(video_clip)
            all_video.append(video_clip)

        all_video = torch.stack(all_video, dim=0)
        return Example({Modalities.VIDEO: all_video, EXAMPLE_INDEX_KEY: idx})

    def _get_clip_timepoints(
        self,
        clip_sampler: ConstantClipsPerVideoSampler,
        duration: float,
    ) -> List[Tuple[float, float]]:
        """Calculate the start and end timepoints for each video clip.

        Parameters
        ----------
        clip_sampler
            The clip sampler instance.
        duration : int
            Total duration of the video.

        Returns
        -------
        list of tuples
            List of tuples containing start and end timepoints of each clip.
        """
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(
                end,
                duration,
                annotation=None,
            )
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints
