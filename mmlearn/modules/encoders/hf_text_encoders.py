"""Huggingface text encoder model."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from hydra_zen import store
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from transformers import AutoModelForTextEncoding
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality


if TYPE_CHECKING:
    from peft import PeftConfig


@store(group="modules/encoders", provider="mmlearn")
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
        model_config_kwargs["use_return_dict"] = True
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

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> BaseModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The `input_ids` will be expected under the `Modalities.TEXT`
            key.

        Returns
        -------
        BaseModelOutput
            The output of the model, including the last hidden state, all hidden states,
            and the attention weights, if `output_attentions` is set to `True`.
        """
        outputs = self.model(
            input_ids=inputs[Modalities.TEXT],
            attention_mask=inputs.get("attention_mask"),
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
