"""Huggingface text encoder models."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from hydra_zen import store
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from transformers import AutoModelForTextEncoding, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core import Modalities


if TYPE_CHECKING:
    from peft import PeftConfig


def freeze_model_layers(
    model: PreTrainedModel,
    freeze_layers: Union[int, float, List[int], bool],
    freeze_layer_norm: bool = True,
) -> None:
    """Freeze model layers based on configuration.

    Parameters
    ----------
    model : PreTrainedModel
        The model whose layers to freeze.
    freeze_layers : Union[int, float, List[int], bool]
        Specification of which layers to freeze.
    freeze_layer_norm : bool
        Whether to freeze layer normalization layers.
    """
    if isinstance(freeze_layers, bool) and freeze_layers:
        for name, param in model.named_parameters():
            param.requires_grad = (
                (not freeze_layer_norm) if "LayerNorm" in name else False
            )
        return

    if isinstance(freeze_layers, (float, int, list)) and model.config.model_type in [
        "flaubert",
        "xlm",
    ]:
        raise ValueError(
            f"Freezing individual layers is not supported for {model.config.model_type} "
            "models. Please use `freeze_layers=False` or `freeze_layers=True`."
        )

    # Get list of layers
    embeddings = model.embeddings
    encoder = getattr(model, "encoder", None) or getattr(model, "transformer", model)
    encoder_layers = (
        getattr(encoder, "layer", None)
        or getattr(encoder, "layers", None)
        or getattr(encoder, "block", None)
    )

    if encoder_layers is None and hasattr(encoder, "albert_layer_groups"):
        encoder_layers = [
            layer
            for group in encoder.albert_layer_groups
            for layer in group.albert_layers
        ]

    modules = [embeddings]
    if encoder_layers is not None and isinstance(encoder_layers, list):
        modules.extend(encoder_layers)

    if isinstance(freeze_layers, float):
        freeze_layers = int(freeze_layers * len(modules))
    if isinstance(freeze_layers, int):
        freeze_layers = list(range(freeze_layers))

    if isinstance(freeze_layers, list):
        for idx, module in enumerate(modules):
            if idx in freeze_layers:
                for name, param in module.named_parameters():
                    param.requires_grad = (
                        (not freeze_layer_norm) if "LayerNorm" in name else False
                    )


class TextEncoderBase(nn.Module):
    """Base class for text encoders."""

    def _get_attention_mask(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get attention mask from inputs."""
        return inputs.get(
            "attention_mask", inputs.get(Modalities.TEXT.attention_mask, None)
        )


@store(group="modules/encoders", provider="mmlearn", hydra_convert="object")
class HFTextEncoder(TextEncoderBase):
    """Wrapper around huggingface models in the `AutoModelForTextEncoding` class."""

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        pooling_layer: Optional[nn.Module] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model."""
        super().__init__()
        if model_config_kwargs is None:
            model_config_kwargs = {}
        model_config_kwargs["output_hidden_states"] = True
        model_config_kwargs["add_pooling_layer"] = False

        model = hf_utils.load_huggingface_model(
            AutoModelForTextEncoding,
            model_name_or_path,
            load_pretrained_weights=pretrained,
            model_config_kwargs=model_config_kwargs,
        )

        if hasattr(model.config, "is_decoder") and model.config.is_decoder:
            raise ValueError("Model is a decoder. Only encoder models are supported.")

        if not pretrained and freeze_layers:
            rank_zero_warn(
                "Freezing layers when loading a model with random weights may lead to "
                "unexpected behavior. Consider setting `freeze_layers=False` if "
                "`pretrained=False`."
            )

        freeze_model_layers(model, freeze_layers, freeze_layer_norm)

        if peft_config is not None:
            model = hf_utils._wrap_peft_model(model, peft_config)

        self.model = model
        self.pooling_layer = pooling_layer

    def forward(self, inputs: Dict[str, Any]) -> BaseModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input data. The `input_ids` will be expected under the `Modalities.TEXT`
            key.

        Returns
        -------
        BaseModelOutput
            The output of the model, including the last hidden state, all hidden states,
            and the attention weights, if `output_attentions` is set to `True`.
        """
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT.name],
            attention_mask=self._get_attention_mask(inputs),
            position_ids=inputs.get("position_ids"),
            output_attentions=inputs.get("output_attentions"),
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        if self.pooling_layer:
            last_hidden_state = self.pooling_layer(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
