import json
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class LabelMapping:
    """
    A simplified label mapping class that provides bidirectional mapping functionality.

    This class maintains two mappings:
    1. Forward mapping (name -> label): Maps semantic names to label values.
    2. Reverse mapping (label -> name): Maps label values back to semantic names.

    The mapping ensures uniqueness of both names and labels.
    """

    def __init__(self, name_to_label: Optional[Dict[str, int]] = None):
        # Forward mapping: name -> label
        self._name_to_label = name_to_label or {}
        # Reverse mapping: label -> name
        self._label_to_name = {v: k for k, v in self._name_to_label.items()}

    def __setitem__(self, name: str, label: int) -> None:
        """
        Set both forward and reverse mappings.

        Args:
            name: Semantic name for the label.
            label: Corresponding label value.

        Raises:
            ValueError: If the label already exists with a different name.
        """
        # Raise error if label exists with different name
        if label in self._label_to_name and self._label_to_name[label] != name:
            raise ValueError(f"Label {label} already exists")

        # Remove old reverse mapping if name exists
        if name in self._name_to_label:
            old_label = self._name_to_label[name]
            del self._label_to_name[old_label]

        self._name_to_label[name] = label
        self._label_to_name[label] = name

    def __getitem__(self, name: str) -> int:
        """
        Get label value by semantic name.

        Args:
            name: Semantic name.

        Returns:
            int: Corresponding label value.

        Raises:
            KeyError: If name is not found in the mapping.
        """
        return self._name_to_label[name]

    def __getattr__(self, name: str) -> int:
        """
        Access label value as an attribute (e.g., mapping.liver).

        Args:
            name: Semantic name.

        Returns:
            int: Corresponding label value.

        Raises:
            AttributeError: If name is not found in the mapping.
        """
        if name in self._name_to_label:
            return self._name_to_label[name]
        raise AttributeError(f"'{name}' not found in label mapping.")

    def __str__(self) -> str:
        """
        Return string representation of the mapping.

        Returns:
            str: String representation of the name_to_label dictionary.
        """
        return str(self._name_to_label)

    def to_json(self) -> str:
        """
        Convert the mapping to a JSON string.

        Returns:
            str: JSON string representation of the mapping.
        """
        return json.dumps(self._name_to_label)

    @classmethod
    def from_json(cls, json_str: str) -> "LabelMapping":
        """
        Initialize a label mapping from a JSON string.

        Args:
            json_str: JSON string representation of the mapping.

        Returns:
            LabelMapping: A new instance initialized with data from JSON.
        """
        json_dict = json.loads(json_str)
        instance = cls(name_to_label=json_dict)
        return instance

    def inverse(self, label: int) -> str:
        """
        Get semantic name by label value (reverse mapping).

        Args:
            label: Label value.

        Returns:
            str: Corresponding semantic name.

        Raises:
            KeyError: If label is not found in the mapping.
        """
        return self._label_to_name[label]

    def __call__(self, name: str) -> int:
        """
        Get label value by semantic name (forward mapping).

        Args:
            name: Semantic name.

        Returns:
            int: Corresponding label value.

        Raises:
            KeyError: If name is not found in the mapping.
        """
        return self._name_to_label[name]

    def has_label(self, label: int) -> bool:
        """
        Check if a label exists in the mapping.

        Args:
            label: Label value to check.

        Returns:
            bool: True if the label exists, False otherwise.
        """
        return label in self._label_to_name

    def items(self) -> Iterator[Tuple[str, int]]:
        """
        Get an iterator of name-label pairs.

        Returns:
            Iterator[Tuple[str, int]]: Iterator of (name, label) pairs.
        """
        return self._name_to_label.items()

    def __iter__(self) -> Iterator[str]:
        """
        Get an iterator of semantic names.

        Returns:
            Iterator[str]: Iterator of semantic names.
        """
        return iter(self._name_to_label)

    def __len__(self) -> int:
        """
        Get the number of mappings.

        Returns:
            int: Number of name-label pairs in the mapping.
        """
        return len(self._name_to_label) 
    

    def __repr__(self) -> str:
        """
        Return string representation of the mapping.

        Returns:
            str: String representation of the name_to_label dictionary.
        """
        return str(self._name_to_label)