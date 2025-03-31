"""PathVQA and VQARAD datasets for medical visual question answering."""

import json
import os
import warnings
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Grayscale, Resize, ToTensor

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


class MedVQA(Dataset[Example]):
    """Module to load PathVQA and VQARAD datasets.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset.
    split : {"train", "val", "test"}, default = "train"
        Split of the dataset to use.
    encoder : dict
        images_filename : str
            Path to .pkl file containing the encoder's input images,
            relative to root_dir.
        image_size : int
            Size of the input images; e.g. 224 for clipvision
        feat_dim : int
            Dimension of the output embedding; e.g. 512 for clipvision
    autoencoder : dict
        available : boolean {True, False}
            Whether or not to return autoencoder images.
        images_filename : str
            Path to .pkl file containing the autoencoder's input images,
            relative to root_dir.
        image_size : int
            Size of the input images; e.g. 128
        feat_dim : str
            Dimension of the output embedding; e.g. 64
    num_ans_candidates : int
        Number of all unique answers in the dataset.
    rgb_transform : Optional[Callable[[Image.Image], torch.Tensor]]
        Transform applied to images that will be passed to the visual encoder.
    ae_transform : Optional[Callable[[Image.Image], torch.Tensor]]
        Transform applied to images that will be passed to the autoencoder.
    tokenizer : Optional[torch.nn.Module]
        Function to tokenize the questions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val", "test"],
        encoder: dict[str, Any],
        autoencoder: dict[str, Any],
        num_ans_candidates: int,
        rgb_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        ae_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialize the dataset."""
        super(MedVQA, self).__init__()
        assert split in ["train", "val", "test"]
        self.autoencoder = autoencoder["available"]
        self.num_ans_candidates = num_ans_candidates

        # transform for encoder images
        if rgb_transform is None:
            self.rgb_transform = ToTensor()
        else:
            self.rgb_transform = rgb_transform

        # transform for autoencoder images
        if self.autoencoder and ae_transform is None:
            self.ae_transform = Compose(
                [
                    Grayscale(1),
                    Resize(autoencoder["image_size"]),
                    CenterCrop(autoencoder["image_size"]),
                    ToTensor(),
                ]
            )
        elif self.autoencoder:
            self.ae_transform = ae_transform

        # tokenizer for textual questions
        self.tokenize_fn = tokenizer

        # load entries
        with open(
            os.path.join(root_dir, "cache", f"{split}_data.json"), encoding="utf-8"
        ) as file:
            self.entries = json.load(file)

        if self.autoencoder:
            self.v_dim = encoder["feat_dim"] + autoencoder["feat_dim"]
        else:
            self.v_dim = encoder["feat_dim"]

    def __getitem__(self, index: int) -> Example:
        """Return an example/sample of the data.

        Returns
        -------
        Example : Example
            One sample of the dataset.
            Modalities.TEXT : torch.Tensor
                Question as tokens.
            Modalities.RGB : torch.Tensor | list[torch.Tensor]
                Preprocessed image.
                A list of two torch Tensors if `autoencoder.available` is set
                True in the dataset config, otherwise a single torch Tensor.
            Modalities.RGB.target : torch.Tensor
                Multi-hot-encoding of the correct answer classes as a vector.
            EXAMPLE_INDEX_KEY : int
                Sample index.
            "qid" : int
                The qid of the sample.
            "answer_type" : str {"yes/no", "number", "OPEN", "CLOSED", ...}
                Answer type.
            "question_type" : str {"what", "does", "are", "SIZE", "PRES", ...}
                Question type.
            "phrase_type" : str {"freeform", "frame"} | int {-1}
                Phrase type.
                (-1 in case the dataset does not have phrase_type info).
            "raw_question" : str
                Question as text.

        Notes
        -----
        If `autoencoder.available` is set True in the dataset configs, a list
        of two torch Tensors are returned as `Modalities.RGB`. The first element
        of the list is the processed image meant for the visual encoder and
        the second element is the image meant for the autoencoder in the MEVF
        pipeline (see [1] for more information). If `autoencoder.available` is
        False, only the image meant for the encoder is returned.

        References
        ----------
        [1] Nguyen, Binh D., Thanh-Toan Do, Binh X. Nguyen, Tuong Do, Erman
        Tjiputra, and Quang D. Tran. "Overcoming data limitation in medical
        visual question answering." In Medical Image Computing and Computer
        Assisted Intervention, MICCAI 2019: 22nd International Conference.
        """
        entry = self.entries[index]
        question = (
            self.tokenize_fn(entry["question"])
            if self.tokenize_fn
            else entry["question"]
        )
        answer = entry["answer"]

        # prepare encoder image
        images_data = Image.open(entry["image_path"]).convert("RGB")
        enc_images_data = self.rgb_transform(images_data)

        # prepare autoencoder image
        if self.autoencoder:
            ae_images_data = self.ae_transform(images_data)

        # pack images if needed
        if self.autoencoder:
            image_data = [enc_images_data, ae_images_data]
        else:
            image_data = enc_images_data

        example = Example(
            {
                Modalities.TEXT.name: question,
                Modalities.RGB.name: image_data,
                EXAMPLE_INDEX_KEY: index,
                "qid": entry["qid"],
                "answer_type": entry["answer_type"],
                "question_type": entry["question_type"],
                "phrase_type": entry["phrase_type"],
                "raw_question": entry["question"],
            },
        )

        if answer is not None:
            labels = answer["labels"]
            scores = answer["scores"]
            target = torch.zeros(self.num_ans_candidates)
            if len(labels):
                labels = torch.from_numpy(np.array(answer["labels"]))
                scores = torch.from_numpy(np.array(answer["scores"], dtype=np.float32))
                target.scatter_(0, labels, scores)
            example[Modalities.RGB.target] = target

        return example

    def __len__(self) -> int:
        """Return size of the dataset."""
        return len(self.entries)


class MedVQAProcessor:
    """Preprocessor for textual reports of MedVQA datasets."""

    def __call__(self, sentence: Union[str, list[str]]) -> Union[str, list[str]]:
        """Process the textual captions."""
        if not isinstance(sentence, (list, str)):
            raise TypeError(
                f"Expected sentence to be a string or list of strings, got {type(sentence)}"
            )

        def _preprocess_sentence(sentence: str) -> str:
            sentence = sentence.lower()
            if "? -yes/no" in sentence:
                sentence = sentence.replace("? -yes/no", "")
            if "? -open" in sentence:
                sentence = sentence.replace("? -open", "")
            if "? - open" in sentence:
                sentence = sentence.replace("? - open", "")
            return (
                sentence.replace(",", "")
                .replace("?", "")
                .replace("'s", " 's")
                .replace("...", "")
                .replace("x ray", "x-ray")
                .replace(".", "")
            )

        if isinstance(sentence, str):
            return _preprocess_sentence(sentence)

        for i, s in enumerate(sentence):
            sentence[i] = _preprocess_sentence(s)

        return sentence
