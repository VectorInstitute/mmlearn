"""Custom transforms for datasets."""

from typing import List, Union

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
