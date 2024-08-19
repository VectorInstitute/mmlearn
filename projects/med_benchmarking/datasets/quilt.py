"""Quilt-1M Dataset."""

import ast
import os
from typing import Callable, List, Literal, Optional

import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("QUILT_ROOT_DIR", MISSING))
class Quilt(Dataset[Example]):
    """Quilt-1M dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset.
    split : {"train", "val"}
        Dataset split.
    subset : List[str], optional, default=["openpath", "pubmed", "quilt", "laion"]
        Subsets of Quilt-1M to load.
    transform : Optional[Callable]
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    processor : Optional[Callable], default=None
        Function applied to the image-target pair.

    Notes
    -----
    If `processor` is not None, it overrides `transform` and `tokenizer`.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"] = "train",
        subset: Optional[List[str]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[Callable[[str], torch.Tensor]] = None,
        processor: Optional[
            Callable[[Image.Image, str], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        # input validation
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Root directory is not accessible: {root_dir}.")

        all_splits = ["train", "val"]
        if split not in all_splits:
            raise ValueError(
                f"Split {split} is not available. Valid splits are {all_splits}."
            )

        all_subsets = ["openpath", "pubmed", "quilt", "laion"]
        if subset is None:
            subset = all_subsets
        for subset_name in subset:
            if subset_name not in all_subsets:
                raise ValueError(
                    f"Subset {subset_name} is not available. Valid subsets are {all_subsets}."
                )

        for func_name, func in zip(
            ["transform", "tokenizer", "processor"], [transform, tokenizer, processor]
        ):
            if func is not None and not callable(func):
                raise ValueError(f"`{func_name}` is not callable.")

        # read entries
        self.data_df = pd.read_csv(os.path.join(root_dir, "quilt_1M_entries.csv"))
        # drop unnecessary and space-consuming columns
        self.data_df.drop(
            columns=[
                "noisy_text",
                "corrected_text",
                "med_umls_ids",
                "roi_text",
                "Unnamed: 0",
            ],
            inplace=True,
        )
        # filter entries based on `split` and `subset`
        self.data_df = self.data_df.loc[
            self.data_df.apply(
                lambda row: row["split"] == split and row["subset"] in subset, axis=1
            )
        ]

        # the 'pathology' column is a list of strings
        self.data_df["pathology"] = self.data_df["pathology"].apply(_safe_eval)

        self.root_dir = root_dir
        self.subset = subset

        if processor is None and transform is None:
            self.transform = Compose([Resize(224), CenterCrop(224), ToTensor()])
        elif processor is None:
            self.transform = transform
        else:
            self.transform = None

        if processor is None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None

        self.processor = processor

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample.

        If a tokenizer is not defined by `processor` or `tokenizer`, only the
        image and free text caption are returned. Otherwise, the image, free-
        text caption, and caption tokens are returned.
        """
        try:
            with Image.open(
                os.path.join(
                    self.root_dir, "quilt_1m", self.data_df["image_path"].iloc[idx]
                )
            ) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"ERROR: {e} on {self.data_df['image_path'].iloc[idx]}")

        if self.transform is not None:
            image = self.transform(image)

        caption = self.data_df["caption"].iloc[idx]
        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        if self.processor is not None:
            image, tokens = self.processor(image, caption)

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: caption,
                EXAMPLE_INDEX_KEY: idx,
                "qid": self.data_df.index[idx],
                "magnification": self.data_df["magnification"].iloc[idx],
                "height": self.data_df["height"].iloc[idx],
                "width": self.data_df["width"].iloc[idx],
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in tokens
                ), f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_df.index)


def _safe_eval(x: str) -> list[str]:
    """Safely evaluate a string as a list."""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)  # type: ignore[no-any-return]
    except (ValueError, SyntaxError):
        return []
