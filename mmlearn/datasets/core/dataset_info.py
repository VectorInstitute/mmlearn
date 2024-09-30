"""Saving the necessary information about a dataset for evaluation tasks."""

from typing import Any, Dict, Optional


class DatasetInfo:
    """

    A class used to represent information about a dataset.

    The DatasetInfo class contains information such as the count of classes,
    label mappings, and label embeddings. It provides getter and setter methods
    to access and modify this information.

    Attributes
    ----------
    _class_count : int
        An integer representing the count of classes.
    _label_mapping : dict
        A dictionary for storing label mappings.
    _label_embedding : dict
        A dictionary for storing label embeddings.
    """

    def __init__(
        self,
        class_count: int = 0,
        label_mapping: Optional[Dict[str, Any]] = None,
        label_embedding: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Parameters

        class_count : int
            An integer representing the count of classes.
        label_mapping : dict
            A dictionary for storing label mappings.
        label_embedding : dict
            A dictionary for storing label embeddings.
        """
        self._class_count = class_count
        self._label_mapping = label_mapping if label_mapping is not None else {}
        self._label_embedding = label_embedding if label_embedding is not None else {}
        self._name = name

    def get_class_count(self) -> int:
        """
        Return the class count.

        Returns
        -------
        int
            The count of classes.
        """
        return self._class_count

    def get_label_mapping(self) -> Dict[str, Any]:
        """
        Return the label mapping.

        Returns
        -------
        dict
            The dictionary of label mappings.
        """
        return self._label_mapping

    def get_label_embedding(self) -> Dict[str, Any]:
        """
        Return the label embedding.

        Returns
        -------
        dict
            The dictionary of label embeddings for each kind of dataset.
            The key value of each element is the name of
            the dataset and its value is the labels embeddings.
        """
        return self._label_embedding

    def set_label_embedding(self, embedding: Dict[str, Any]) -> None:
        """
        Set the label embedding.

        Parameters
        ----------
        embedding : dict
            A dictionary for label embeddings.
        """
        self._label_embedding = embedding

    def get_name(self) -> str:
        """
        Return the label embedding.

        Returns
        -------
        str
            name
        """
        return self._name or ""
