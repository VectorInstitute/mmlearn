"""Module for managing supported modalities in the library."""

import re
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import Self


_default_supported_modalities = ["rgb", "depth", "thermal", "text", "audio", "video"]


class Modality(str):
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

    Attributes
    ----------
    value : str
        The name of the modality.
    properties : dict[str, str]
        The properties associated with the modality. By default, the properties are
        `target`, `mask`, `embedding`, `masked_embedding`, and `ema_embedding`.
        These default properties apply to all newly created modality types
        automatically. Modality-specific properties can be added using the
        `add_property` method or by passing them as a dictionary to the constructor.
    """

    _default_properties = {
        "target": "{}_target",
        "mask": "{}_mask",
        "embedding": "{}_embedding",
        "masked_embedding": "{}_masked_embedding",
        "ema_embedding": "{}_ema_embedding",
    }

    if TYPE_CHECKING:

        def __getattr__(self, attr: str) -> Any:
            """Get the value of the attribute."""
            ...

        def __setattr__(self, attr: str, value: Any) -> None:
            """Set the value of the attribute."""
            ...

    def __new__(
        cls, name: str, modality_specific_properties: Optional[dict[str, str]] = None
    ) -> Self:
        """Initialize the modality with the name and properties."""
        instance = super(Modality, cls).__new__(cls, name.lower())
        properties = cls._default_properties.copy()
        if modality_specific_properties is not None:
            properties.update(modality_specific_properties)
        instance._properties = properties

        for property_name, format_string in instance._properties.items():
            instance._set_property_as_attr(property_name, format_string)

        return instance

    @property
    def value(self) -> str:
        """Return the name of the modality."""
        return self.__str__()

    @property
    def properties(self) -> dict[str, str]:
        """Return the properties associated with the modality."""
        return {name: getattr(self, name) for name in self._properties}

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

        Raises
        ------
        ValueError
            If the property already exists for the modality or if the format string is
            invalid.
        """
        if name in self._properties:
            raise ValueError(
                f"Property '{name}' already exists for modality '{super().__str__()}'."
            )
        self._properties[name] = format_string
        self._set_property_as_attr(name, format_string)

    def _set_property_as_attr(self, name: str, format_string: str) -> None:
        """Set the property as an attribute of the modality."""
        if not _is_format_string(format_string):
            raise ValueError(
                f"Invalid format string '{format_string}' for property "
                f"'{name}' of modality '{super().__str__()}'."
            )
        setattr(self, name, format_string.format(self.value))

    def __str__(self) -> str:
        """Return the object as a string."""
        return self.lower()

    def __repr__(self) -> str:
        """Return the string representation of the modality."""
        return f"<Modality: {self.upper()}>"

    def __hash__(self) -> int:
        """Return the hash of the modality name and properties."""
        return hash((self.value, tuple(self._properties.items())))

    def __eq__(self, other: object) -> bool:
        """Check if two modality types are equal.

        Two modality types are equal if they have the same name and properties.
        """
        return isinstance(other, Modality) and (
            (self.__str__() == other.__str__())
            and (self._properties == other._properties)
        )


class ModalityRegistry:
    """Modality registry.

    A singleton class that manages the supported modalities (and their properties) in
    the library. The class provides methods to add new modalities and properties, and
    to access the existing modalities. The class is implemented as a singleton to
    ensure that there is only one instance of the registry in the library.
    """

    _instance = None

    def __new__(cls) -> Self:
        """Create a new instance of the class if it does not exist."""
        if cls._instance is None:
            cls._instance = super(ModalityRegistry, cls).__new__(cls)
            cls._instance._modality_registry = {}  # type: ignore[attr-defined]
            for modality in _default_supported_modalities:
                cls._instance.register_modality(modality)
        return cls._instance

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

        Raises
        ------
        ValueError
            If the modality already exists in the registry.
        """
        if name.lower() in self._modality_registry:
            raise ValueError(f"Modality '{name}' already exists in the registry.")

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

        Raises
        ------
        ValueError
            If the property already exists for the default properties or if the format
            string is invalid.
        """
        for modality in self._modality_registry.values():
            modality.add_property(name, format_string)

        # add the property to the default properties for new modalities
        Modality._default_properties[name.lower()] = format_string

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
        return self._modality_registry[name.lower()]  # type: ignore[index,return-value]

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
            return self._modality_registry[name.lower()]  # type: ignore[index,return-value]
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
