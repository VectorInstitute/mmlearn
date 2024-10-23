"""Test modality registry and related classes."""

import pytest

from mmlearn.datasets.core.modalities import Modality, ModalityRegistry


def test_modality_init():
    """Test modality initialization."""
    modality = Modality("text")
    assert modality.name == "text"
    assert modality.target == "text_target"
    assert modality.mask == "text_mask"
    assert modality.embedding == "text_embedding"
    assert modality.masked_embedding == "text_masked_embedding"
    assert modality.ema_embedding == "text_ema_embedding"

    for name, property_name in modality.properties.items():
        assert getattr(modality, name) == property_name

    assert str(modality) == "text"


def test_modality_init_with_custom_properties():
    """Test modality initialization with custom properties."""
    modality = Modality("text", {"a_custom_property": "{}_custom_value"})
    assert modality.a_custom_property == "text_custom_value"
    assert "a_custom_property" in modality.properties


def test_modality_add_property():
    """Test modality add property."""
    modality = Modality("text")
    modality.add_property("a_custom_property", "{}_custom_value")
    assert modality.a_custom_property == "text_custom_value"

    with pytest.warns(UserWarning):
        modality.add_property("a_custom_property", "{}_custom_value2")
    assert modality.a_custom_property == "text_custom_value2"

    with pytest.raises(ValueError):
        modality.add_property("a_custom_property_2", "custom_value")


def test_modality_registry_singleton():
    """Test modality registry singleton."""
    registry1 = ModalityRegistry()
    registry2 = ModalityRegistry()
    assert registry1 is registry2


def test_modality_registration():
    """Test modality registration."""
    registry = ModalityRegistry()
    registry.register_modality("modality_name")
    assert "modality_name" in registry._modality_registry
    assert registry.has_modality("modality_name")

    modality = registry.get_modality("modality_name")
    assert isinstance(modality, Modality)
    assert modality.name == "modality_name"
    assert sorted(modality.properties) == sorted(
        [
            "target",
            "mask",
            "embedding",
            "masked_embedding",
            "ema_embedding",
            "attention_mask",
        ]
    )

    # check attribute access
    assert registry.modality_name.target == "modality_name_target"
    assert registry.modality_name.attention_mask == "modality_name_attention_mask"
    assert registry.modality_name.mask == "modality_name_mask"
    assert registry.modality_name.embedding == "modality_name_embedding"
    assert registry.modality_name.masked_embedding == "modality_name_masked_embedding"
    assert registry.modality_name.ema_embedding == "modality_name_ema_embedding"

    # check list_modalities
    assert registry.list_modalities()[-1] == modality

    # check registration with same name overwrites the existing modality
    with pytest.warns(UserWarning):
        registry.register_modality(
            "modality_name", {"a_custom_property": "{}_custom_value"}
        )
    assert registry.modality_name.a_custom_property == "modality_name_custom_value"

    # check addition of modality specific properties


def test_modality_registration_with_custom_properties():
    """Test modality registration with custom properties."""
    registry = ModalityRegistry()
    registry.register_modality(
        "modality_name_2", {"a_custom_property": "{}_custom_value"}
    )
    modality = registry.get_modality("modality_name_2")
    assert modality.name == "modality_name_2"
    assert modality.a_custom_property == "modality_name_2_custom_value"


def test_modality_registration_with_invalid_custom_properties():
    """Test modality registration with invalid custom properties."""
    registry = ModalityRegistry()
    with pytest.raises(ValueError):
        registry.register_modality(
            "modality_name_3", {"invalid_property": "invalid_value"}
        )
