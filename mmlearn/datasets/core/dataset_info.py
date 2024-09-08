"""Saving the necessary imformstion about a dataset for. evaluation tasks"""

class DatasetInfo:
    def __init__(self, class_count=0, label_mapping=None, label_embedding=None):
        """
        Parameters:
        ----------
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

    # Getter for class_count
    def get_class_count(self):
        return self._class_count

    # Setter for class_count
    def set_class_count(self, count):
        if isinstance(count, int) and count >= 0:
            self._class_count = count
        else:
            raise ValueError("class_count must be a non-negative integer")

    # Getter for label_mapping
    def get_label_mapping(self):
        return self._label_mapping

    # Setter for label_mapping
    def set_label_mapping(self, mapping):
        if isinstance(mapping, dict):
            self._label_mapping = mapping
        else:
            raise ValueError("label_mapping must be a dictionary")

    # Getter for label_embedding
    def get_label_embedding(self):
        return self._label_embedding

    # Setter for label_embedding
    def set_label_embedding(self, embedding):
        self._label_embedding = embedding
