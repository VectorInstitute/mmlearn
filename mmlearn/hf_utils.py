"""Utilities for loading components from the HuggingFace `transformers` library."""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from lightning_utilities.core.imports import RequirementCache
from torch import nn
from transformers.models.auto.auto_factory import AutoConfig, _BaseAutoModelClass


logger = logging.getLogger(__name__)

_PEFT_AVAILABLE = RequirementCache("peft>=0.12.0")


if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel


def load_huggingface_model(
    model_type: Type[_BaseAutoModelClass],
    model_name_or_path: str,
    load_pretrained_weights: bool = True,
    get_model_attr: Optional[str] = None,
    model_config_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Load a model from the HuggingFace `transformers` library.

    Parameters
    ----------
    model_type : Type[_BaseAutoModelClass]
        The model class to instantiate e.g. `AutoModel`.
    model_name_or_path : str
        The model name or path to load the model from.
    load_pretrained_weights : bool, optional, default=True
        Whether to load the pretrained weights or not. If false, the argument
        `pretrained_model_name_or_path` will be used to get the model configuration
        and the model will be initialized with random weights.
    get_model_attr : str, optional, default=None
        If not None, the attribute of the model to return. For example, if the model
        is an `AutoModel` and `get_model_attr='encoder'`, the encoder part of the
        model will be returned. If None, the full model will be returned.
    **model_config_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to the model configuration.
        The values in kwargs of any keys which are configuration attributes will
        be used to override the loaded values. Behavior concerning key/value pairs
        whose keys are *not* configuration attributes is controlled by the
        `return_unused_kwargs` keyword parameter.

    Returns
    -------
    nn.Module
        The instantiated model.
    """
    model_config_kwargs = model_config_kwargs or {}
    if load_pretrained_weights:
        model = model_type.from_pretrained(model_name_or_path, **model_config_kwargs)
    else:
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            return_unused_kwargs=True,
            **model_config_kwargs,
        )
        model = model_type.from_config(config, **kwargs)

    if get_model_attr is not None and hasattr(model, get_model_attr):
        model = getattr(model, get_model_attr)

    return model


def _wrap_peft_model(model: nn.Module, peft_config: "PeftConfig") -> "PeftModel":
    """Wrap the model with the `peft` library for parameter-efficient finetuning."""
    if not _PEFT_AVAILABLE:
        raise ModuleNotFoundError(str(_PEFT_AVAILABLE))
    from peft import get_peft_model

    peft_model = get_peft_model(model, peft_config)
    trainable_params, all_param = peft_model.get_nb_trainable_parameters()

    logger.info(
        f"Parameter-efficient finetuning {peft_model.base_model.model.__class__.__name__} "
        f"with {trainable_params:,d} trainable parameters out of {all_param:,d} total parameters. "
        f"Trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return peft_model
