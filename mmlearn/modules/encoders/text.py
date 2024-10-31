"""Huggingface text encoder models."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from hydra_zen import store
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from transformers import AutoModel, AutoModelForTextEncoding
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core import Modalities


if TYPE_CHECKING:
    from peft import PeftConfig


@store(group="modules/encoders", provider="mmlearn", hydra_convert="object")
class HFTextEncoder(nn.Module):
    """Wrapper around huggingface models in the `AutoModelForTextEncoding` class.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or a local path from which to load the model.
    pretrained : bool, default=True
        Whether to load the pretrained weights or not.
    pooling_layer : nn.Module, optional, default=None
        Pooling layer to apply to the last hidden state of the model.
    freeze_layers : int | float | List[int] | bool, default=False
        Whether to freeze layers of the model and which layers to freeze. If `True`,
        all model layers are frozen. If it is an integer, the first `N` layers of
        the model are frozen. If it is a float, the first `N` percent of the layers
        are frozen. If it is a list of integers, the layers at the indices in the
        list are frozen.
    freeze_layer_norm : bool, default=True
        Whether to freeze the layer normalization layers of the model.
    peft_config : PeftConfig, optional, default=None
        The configuration from the `peft` library to use to wrap the model
        for parameter-efficient finetuning.
    model_config_kwargs : Dict[str, Any], optional, default=None
        Additional keyword arguments to pass to the model configuration.

    Warns
    -----
    UserWarning
        If both `peft_config` and `freeze_layers` are set. The `peft_config` will
        override the `freeze_layers` setting.


    """

    def __init__(  # noqa: PLR0912
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
                "`pretrained=False`.",
            )

        if isinstance(freeze_layers, bool) and freeze_layers:
            for name, param in model.named_parameters():
                param.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in name else False
                )

        if isinstance(
            freeze_layers, (float, int, list)
        ) and model.config.model_type in ["flaubert", "xlm"]:
            # flaubert and xlm models have a different architecture that does not
            # support freezing individual layers in the same way as other models
            raise ValueError(
                f"Freezing individual layers is not supported for {model.config.model_type} "
                "models. Please use `freeze_layers=False` or `freeze_layers=True`."
            )

        # get list of layers
        embeddings = model.embeddings
        encoder = getattr(model, "encoder", None) or getattr(
            model, "transformer", model
        )
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
            attention_mask=inputs.get(
                "attention_mask", inputs.get(Modalities.TEXT.attention_mask, None)
            ),
            position_ids=inputs.get("position_ids"),
            output_attentions=inputs.get("output_attentions"),
            return_dict=True,
        )
        last_hidden_state = outputs.hidden_states[-1]  # NOTE: no layer norm applied
        if self.pooling_layer:
            last_hidden_state = self.pooling_layer(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RegressionHead(nn.Module):
    """Regression head for Data2Vec text encoder.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension.
    num_layers : int
        Number of layers in the regression head.
    """

    def __init__(self, embed_dim: int, num_layers: int = 1) -> None:
        """Initialize the regression head."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        curr_dim = embed_dim

        for i in range(num_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            layers.extend([nn.Linear(curr_dim, next_dim), nn.GELU()])
            curr_dim = next_dim

        layers.append(nn.Linear(curr_dim, embed_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through the regression head.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.layers(x)


@store(group="modules/encoders", provider="mmlearn", hydra_convert="object")
class Data2VecTextEncoder(nn.Module):
    """Text encoder for Data2Vec implementation.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or path.
    head_layers : int
        Number of layers in regression head.
    pretrained : bool
        Whether to use pretrained weights.
    freeze_layers : Union[int, float, List[int], bool]
        Which layers to freeze.
    freeze_layer_norm : bool
        Whether to freeze layer norm parameters.
    peft_config : Optional[PeftConfig]
        PEFT configuration for efficient fine-tuning.
    model_kwargs : Optional[Dict[str, Any]]
        Additional model configuration arguments.
    """

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

        # Handle layer freezing
        if isinstance(freeze_layers, bool) and freeze_layers:
            for name, param in self.model.named_parameters():
                param.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in name else False
                )

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
        """Forward pass through the encoder.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Input dictionary containing text inputs under Modalities.TEXT key.
        output_hidden_states : bool
            Whether to return all hidden states.

        Returns
        -------
        BaseModelOutput
            Model outputs including hidden states if requested.
        """
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT.name],
            attention_mask=inputs.get(
                "attention_mask", inputs.get(Modalities.TEXT.attention_mask, None)
            ),
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
        """Get intermediate hidden states with optional normalization.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Input dictionary with text inputs.
        normalize : bool
            Whether to apply any normalization.
        layer_norm : bool
            Whether to apply layer normalization.
        instance_norm : bool
            Whether to apply instance normalization.
        batch_norm : bool
            Whether to apply batch normalization.

        Returns
        -------
        List[torch.Tensor]
            List of hidden states from intermediate layers.
        """
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT.name],
            attention_mask=inputs.get(
                "attention_mask", inputs.get(Modalities.TEXT.attention_mask, None)
            ),
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
