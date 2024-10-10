"""Module for managing supported modalities in the library."""

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from typing_extensions import Self


_DEFAULT_SUPPORTED_MODALITIES = ["rgb", "depth", "thermal", "text", "audio", "video"]


@dataclass
class Modality:
    """Class to represent a modality in the library.

    This class is used to represent a modality in the library. It contains the name of
    the modality and the properties that can be associated with it. The properties are
    dynamically generated based on the name of the modality and can be accessed as
    attributes of the class.

    Parameters
    ----------
    name : str
        The name of the modality.
    modality_specific_properties : Optional[dict[str, str]], optional, default=None
        Additional properties specific to the modality, by default None

    Raises
    ------
    ValueError
        If the property already exists for the modality or if the format string is
        invalid.
    """

    name: str
    target: str = field(init=False, repr=False)
    attention_mask: str = field(init=False, repr=False)
    mask: str = field(init=False, repr=False)
    embedding: str = field(init=False, repr=False)
    masked_embedding: str = field(init=False, repr=False)
    ema_embedding: str = field(init=False, repr=False)
    modality_specific_properties: Optional[dict[str, str]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize the modality with the name and properties."""
        self.name = self.name.lower()
        self._properties = {}

        for field_name in self.__dataclass_fields__:
            if field_name not in ("name", "modality_specific_properties"):
                field_value = f"{self.name}_{field_name}"
                self._properties[field_name] = field_value
                setattr(self, field_name, field_value)

        if self.modality_specific_properties is not None:
            for (
                property_name,
                format_string,
            ) in self.modality_specific_properties.items():
                self.add_property(property_name, format_string)

    @property
    def properties(self) -> dict[str, str]:
        """Return the properties associated with the modality."""
        return self._properties

    def add_property(self, name: str, format_string: str) -> None:
        """Add a new property to the modality.

        Parameters
        ----------
        name : str
            The name of the property.
        format_string : str
            The format string for the property. The format string should contain a
            placeholder that will be replaced with the name of the modality when the
            property is accessed.

        Warns
        -----
        UserWarning
            If the property already exists for the modality. It will overwrite the
            existing property.

        Raises
        ------
        ValueError
            If `format_string` is invalid. A valid format string contains at least one
            placeholder enclosed in curly braces.
        """
        if name in self._properties:
            warnings.warn(
                f"Property '{name}' already exists for modality '{super().__str__()}'."
                "Will overwrite the existing property.",
                category=UserWarning,
                stacklevel=2,
            )

        if not _is_format_string(format_string):
            raise ValueError(
                f"Invalid format string '{format_string}' for property "
                f"'{name}' of modality '{super().__str__()}'."
            )

        self._properties[name] = format_string.format(self.name)
        setattr(self, name, self._properties[name])

    def __str__(self) -> str:
        """Return the object as a string."""
        return self.name.lower()


class ModalityRegistry:
    """Modality registry.

    A singleton class that manages the supported modalities (and their properties) in
    the library. The class provides methods to add new modalities and properties, and
    to access the existing modalities. The class is implemented as a singleton to
    ensure that there is only one instance of the registry in the library.
    """

    _instance: ClassVar[Any] = None
    _modality_registry: dict[str, Modality] = {}

    def __new__(cls) -> Self:
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modality_registry = {}
        return cls._instance  # type: ignore[no-any-return]

    def register_modality(
        self, name: str, modality_specific_properties: Optional[dict[str, str]] = None
    ) -> None:
        """Add a new modality to the registry.

        Parameters
        ----------
        name : str
            The name of the modality.
        modality_specific_properties : Optional[dict[str, str]], optional, default=None
            Additional properties specific to the modality.

        Warns
        -----
        UserWarning
            If the modality already exists in the registry. It will overwrite the
            existing modality.

        """
        if name.lower() in self._modality_registry:
            warnings.warn(
                f"Modality '{name}' already exists in the registry. Overwriting...",
                category=UserWarning,
                stacklevel=2,
            )

        name = name.lower()
        modality = Modality(name, modality_specific_properties)
        self._modality_registry[name] = modality
        setattr(self, name, modality)

    def add_default_property(self, name: str, format_string: str) -> None:
        """Add a new property that is applicable to all modalities.

        Parameters
        ----------
        name : str
            The name of the property.
        format_string : str
            The format string for the property. The format string should contain a
            placeholder that will be replaced with the name of the modality when the
            property is accessed.

        Warns
        -----
        UserWarning
            If the property already exists for the default properties. It will
            overwrite the existing property.

        Raises
        ------
        ValueError
            If the format string is invalid. A valid format string contains at least one
            placeholder enclosed in curly braces.
        """
        for modality in self._modality_registry.values():
            modality.add_property(name, format_string)

    def has_modality(self, name: str) -> bool:
        """Check if the modality exists in the registry.

        Parameters
        ----------
        name : str
            The name of the modality.

        Returns
        -------
        bool
            True if the modality exists in the registry, False otherwise.
        """
        return name.lower() in self._modality_registry

    def get_modality(self, name: str) -> Modality:
        """Get the modality name from the registry.

        Parameters
        ----------
        name : str
            The name of the modality.

        Returns
        -------
        Modality
            The modality object from the registry.
        """
        return self._modality_registry[name.lower()]

    def get_modality_properties(self, name: str) -> dict[str, str]:
        """Get the properties of a modality from the registry.

        Parameters
        ----------
        name : str
            The name of the modality.

        Returns
        -------
        dict[str, str]
            The properties associated with the modality.
        """
        return self.get_modality(name).properties

    def list_modalities(self) -> list[Modality]:
        """Get the list of supported modalities in the registry.

        Returns
        -------
        list[Modality]
            The list of supported modalities in the registry.
        """
        return list(self._modality_registry.values())

    def __getattr__(self, name: str) -> Modality:
        """Access a modality as an attribute by its name."""
        if name.lower() in self._modality_registry:
            return self._modality_registry[name.lower()]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def _is_format_string(string: str) -> bool:
    """Check if the string is a format string.

    A format string is a string that contains one or more placeholders enclosed in
    curly braces. This function checks if the string contains at least one placeholder
    and returns a boolean value indicating whether it is a format string.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a format string, False otherwise.
    """
    pattern = r"\{.*?\}"
    return bool(re.search(pattern, string))


Modalities = ModalityRegistry()

for modality in _DEFAULT_SUPPORTED_MODALITIES:
    Modalities.register_modality(modality)
