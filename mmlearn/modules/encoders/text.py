"""Huggingface text encoder models."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from hydra_zen import store
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from transformers import AutoModel, AutoModelForTextEncoding, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core import Modalities
from mmlearn.modules.layers.mlp import MLP


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


class RegressionHead(nn.Module):
    """Regression head for Data2Vec text encoder."""

    def __init__(self, embed_dim: int, num_layers: int = 1) -> None:
        """Initialize the regression head.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input embeddings
        num_layers : int, optional
            Number of layers in the regression head, by default 1
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            hidden_dims = []
        else:
            hidden_dims = [embed_dim * 2] + [embed_dim * 2] * (num_layers - 2)

        self.layers = MLP(
            in_dim=embed_dim,
            out_dim=embed_dim,
            hidden_dims=hidden_dims,
            activation_layer=nn.GELU,
            norm_layer=None,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.layers(x)


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


@store(group="modules/encoders", provider="mmlearn", hydra_convert="object")
class Data2VecTextEncoder(TextEncoderBase):
    """Text encoder for Data2Vec implementation."""

    def __init__(
        self,
        model_name_or_path: str,
        head_layers: int = 1,
        pretrained: bool = True,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the encoder."""
        super().__init__()

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["output_hidden_states"] = True
        model_kwargs["add_pooling_layer"] = False

        # Load base model
        self.model = (
            AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            if pretrained
            else AutoModel.from_config(
                AutoModel.config_class.from_pretrained(model_name_or_path)
            )
        )

        freeze_model_layers(self.model, freeze_layers, freeze_layer_norm)

        # Build regression head
        self.regression_head = RegressionHead(
            self.model.config.hidden_size, num_layers=head_layers
        )

        if peft_config is not None:
            self.model = hf_utils._wrap_peft_model(self.model, peft_config)

    def forward(
        self,
        inputs: Dict[str, Any],
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Forward pass through the encoder."""
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT.name],
            attention_mask=self._get_attention_mask(inputs),
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        if self.regression_head is not None:
            last_hidden_state = self.regression_head(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def get_hidden_states(
        self,
        inputs: Dict[str, Any],
        normalize: bool = False,
        layer_norm: bool = False,
        instance_norm: bool = False,
        batch_norm: bool = False,
    ) -> List[torch.Tensor]:
        """Get intermediate hidden states with optional normalization."""
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT.name],
            attention_mask=self._get_attention_mask(inputs),
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = list(outputs.hidden_states)

        if normalize:
            if instance_norm or batch_norm:
                hidden_states = [h.permute(0, 2, 1) for h in hidden_states]

                if batch_norm:
                    hidden_states = [
                        F.batch_norm(h.float(), None, None, training=True)
                        for h in hidden_states
                    ]

                if instance_norm:
                    hidden_states = [F.instance_norm(h.float()) for h in hidden_states]

                hidden_states = [h.permute(0, 2, 1) for h in hidden_states]

            if layer_norm:
                hidden_states = [
                    F.layer_norm(h.float(), h.shape[-1:]) for h in hidden_states
                ]

        return hidden_states
