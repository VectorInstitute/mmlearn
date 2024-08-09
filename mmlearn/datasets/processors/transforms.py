"""Custom transforms for datasets."""

from typing import List, Literal, Union

from hydra_zen import store
from timm.data.transforms import ResizeKeepRatio
from torchvision import transforms


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


class MedVQAProcessor:
    """Preprocessor for textual reports of MedVQA datasets."""

    def __call__(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
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


@store(group="datasets/transforms", provider="mmlearn")  # type: ignore[misc]
def med_clip_vision_transform(
    image_crop_size: int = 224, job_type: Literal["train", "eval"] = "train"
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    return transforms.Compose(
        [
            ResizeKeepRatio(512, interpolation="bicubic"),
            transforms.RandomCrop(image_crop_size)
            if job_type == "train"
            else transforms.CenterCrop(image_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
