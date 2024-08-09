"""Wrappers and interfaces for the CLIP models."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import transformers
from hydra_zen import store
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.modules.layers import PatchDropout


if TYPE_CHECKING:
    from peft import PeftConfig


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name_or_path="openai/clip-vit-base-patch16",
)
class HFCLIPTextEncoder(nn.Module):
    """Wrapper around the `CLIPTextModel` from HuggingFace.

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

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        pooling_layer: Optional[nn.Module] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the CLIP text model."""
        super().__init__()
        _warn_freeze_with_peft(peft_config, freeze_layers)

        model = hf_utils.load_huggingface_model(
            transformers.CLIPTextModel,
            model_name_or_path=model_name_or_path,
            load_pretrained_weights=pretrained,
            model_config_kwargs=model_config_kwargs,
        )
        model = _freeze_text_model(model, freeze_layers, freeze_layer_norm)
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


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name_or_path="openai/clip-vit-base-patch16",
)
class HFCLIPVisionEncoder(nn.Module):
    """Wrapper around the `CLIPVisionModel` from HuggingFace.

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
    patch_dropout_rate : float, default=0.0
        The proportion of patch embeddings to drop out.
    patch_dropout_shuffle : bool, default=False
        Whether to shuffle the patches while applying patch dropout.
    patch_dropout_bias : float, optional, default=None
        The bias to apply to the patch dropout mask.
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

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        pooling_layer: Optional[nn.Module] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        patch_dropout_rate: float = 0.0,
        patch_dropout_shuffle: bool = False,
        patch_dropout_bias: Optional[float] = None,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the CLIP vision model."""
        super().__init__()
        _warn_freeze_with_peft(peft_config, freeze_layers)

        model = hf_utils.load_huggingface_model(
            transformers.CLIPVisionModel,
            model_name_or_path=model_name_or_path,
            load_pretrained_weights=pretrained,
            model_config_kwargs=model_config_kwargs,
        )
        model = _freeze_vision_model(model, freeze_layers, freeze_layer_norm)
        if peft_config is not None:
            model = hf_utils._wrap_peft_model(model, peft_config)

        self.model = model.vision_model
        self.pooling_layer = pooling_layer
        self.patch_dropout = None
        if patch_dropout_rate > 0:
            self.patch_dropout = PatchDropout(
                keep_rate=1 - patch_dropout_rate,
                token_shuffling=patch_dropout_shuffle,
                bias=patch_dropout_bias,
            )

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> BaseModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The image tensor will be expected under the `Modalities.RGB`
            key.

        Returns
        -------
        BaseModelOutput
            The output of the model, including the last hidden state, all hidden states,
            and the attention weights, if `output_attentions` is set to `True`.

        """
        # FIXME: handle other vision modalities
        pixel_values = inputs[Modalities.RGB]
        hidden_states = self.model.embeddings(pixel_values)
        if self.patch_dropout is not None:
            hidden_states = self.patch_dropout(hidden_states)
        hidden_states = self.model.pre_layrnorm(hidden_states)

        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=inputs.get(
                "output_attentions", self.model.config.output_attentions
            ),
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs[0]
        if self.pooling_layer is not None:
            last_hidden_state = self.pooling_layer(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name_or_path="openai/clip-vit-base-patch16",
)
class HFCLIPTextEncoderWithProjection(nn.Module):
    """Wrapper around the `CLIPTextModelWithProjection` from HuggingFace.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or a local path from which to load the model.
    pretrained : bool, default=True
        Whether to load the pretrained weights or not.
    use_all_token_embeddings : bool, default=False
        Whether to use all token embeddings for the text. If `False` the first token
        embedding will be used.
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

    Warns
    -----
    UserWarning
        If both `peft_config` and `freeze_layers` are set. The `peft_config` will
        override the `freeze_layers` setting.

    """

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        use_all_token_embeddings: bool = False,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        _warn_freeze_with_peft(peft_config, freeze_layers)

        self.use_all_token_embeddings = use_all_token_embeddings
        model = hf_utils.load_huggingface_model(
            transformers.CLIPTextModelWithProjection,
            model_name_or_path=model_name_or_path,
            load_pretrained_weights=pretrained,
            model_config_kwargs=model_config_kwargs,
        )

        model = _freeze_text_model(model, freeze_layers, freeze_layer_norm)
        if peft_config is not None:
            model = hf_utils._wrap_peft_model(model, peft_config)

        self.model = model

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The `input_ids` will be expected under the `Modalities.TEXT`
            key.

        Returns
        -------
        Tuple[torch.Tensor]
            The text embeddings. Will be a tuple with a single element.
        """
        input_ids = inputs[Modalities.TEXT]
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")

        if self.use_all_token_embeddings:
            text_outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            # TODO: add more options for pooling before projection
            text_embeds = self.model.text_projection(text_outputs.last_hidden_state)
        else:
            text_embeds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            ).text_embeds

        return (text_embeds,)


@store(
    group="modules/encoders",
    provider="mmlearn",
    model_name_or_path="openai/clip-vit-base-patch16",
)
class HFCLIPVisionEncoderWithProjection(nn.Module):
    """Wrapper around the `CLIPVisionModelWithProjection` class from HuggingFace.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or a local path from which to load the model.
    pretrained : bool, default=True
        Whether to load the pretrained weights or not.
    use_all_token_embeddings : bool, default=False
        Whether to use all token embeddings for the text. If `False` the first token
        embedding will be used.
    freeze_layers : int | float | List[int] | bool, default=False
        Whether to freeze layers of the model and which layers to freeze. If `True`,
        all model layers are frozen. If it is an integer, the first `N` layers of
        the model are frozen. If it is a float, the first `N` percent of the layers
        are frozen. If it is a list of integers, the layers at the indices in the
        list are frozen.
    freeze_layer_norm : bool, default=True
        Whether to freeze the layer normalization layers of the model.
    patch_dropout_rate : float, default=0.0
        The proportion of patch embeddings to drop out.
    patch_dropout_shuffle : bool, default=False
        Whether to shuffle the patches while applying patch dropout.
    patch_dropout_bias : float, optional, default=None
        The bias to apply to the patch dropout mask.
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

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        use_all_token_embeddings: bool = False,
        patch_dropout_rate: float = 0.0,
        patch_dropout_shuffle: bool = False,
        patch_dropout_bias: Optional[float] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        _warn_freeze_with_peft(peft_config, freeze_layers)

        self.use_all_token_embeddings = use_all_token_embeddings
        model = hf_utils.load_huggingface_model(
            transformers.CLIPVisionModelWithProjection,
            model_name_or_path=model_name_or_path,
            load_pretrained_weights=pretrained,
            model_config_kwargs=model_config_kwargs,
        )

        model = _freeze_vision_model(model, freeze_layers, freeze_layer_norm)
        if peft_config is not None:
            model = hf_utils._wrap_peft_model(model, peft_config)

        self.model = model
        self.patch_dropout = None
        if patch_dropout_rate > 0:
            self.patch_dropout = PatchDropout(
                keep_rate=1 - patch_dropout_rate,
                token_shuffling=patch_dropout_shuffle,
                bias=patch_dropout_bias,
            )

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The image tensor will be expected under the `Modalities.RGB`
            key.

        Returns
        -------
        Tuple[torch.Tensor]
            The image embeddings. Will be a tuple with a single element.
        """
        pixel_values = inputs[Modalities.RGB]
        hidden_states = self.model.vision_model.embeddings(pixel_values)
        if self.patch_dropout is not None:
            hidden_states = self.patch_dropout(hidden_states)
        hidden_states = self.model.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.model.vision_model.encoder(
            inputs_embeds=hidden_states, return_dict=True
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        if self.use_all_token_embeddings:
            pooled_output = last_hidden_state
        else:
            pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.model.vision_model.post_layernorm(pooled_output)

        return (self.model.visual_projection(pooled_output),)


@store(group="modules/encoders", provider="mmlearn")
class PubMedBERTForCLIPTextEncoding(nn.Module):
    """BiomedNLP's PubMedBERT model for CLIP text encoding.

    This module is wrapper around the PubMedBERT model from huggingface.

    Parameters
    ----------
    pretrained : bool, default=False
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

    def __init__(
        self,
        pretrained: bool = True,
        pooling_layer: Optional[nn.Module] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional["PeftConfig"] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        _warn_freeze_with_peft(peft_config, freeze_layers)

        model = hf_utils.load_huggingface_model(
            transformers.AutoModelForMaskedLM,
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            load_pretrained_weights=pretrained,
            get_model_attr="bert",
            model_config_kwargs=model_config_kwargs,
        )

        if isinstance(freeze_layers, bool) and freeze_layers:
            for name, param in model.named_parameters():
                param.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in name else False
                )

        layers = [model.embeddings, *model.encoder.layer]
        if isinstance(freeze_layers, float):
            freeze_layers = int(freeze_layers * len(layers))
        if isinstance(freeze_layers, int):
            freeze_layers = list(range(freeze_layers))

        if isinstance(freeze_layers, list):
            for idx, layer in enumerate(layers):
                if idx in freeze_layers:
                    for name, param in layer.named_parameters():
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
        output = self.model(
            input_ids=inputs[Modalities.TEXT],
            attention_mask=inputs.get("attention_mask"),
            inputs_embeds=inputs.get("inputs_embeds"),
            output_attentions=inputs.get("output_attentions"),
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = output.last_hidden_state
        if self.pooling_layer is not None:
            last_hidden_state = self.pooling_layer(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


#### Utility methods ####


def _freeze_text_model(
    model: nn.Module,
    freeze_layers: Union[int, float, List[int], bool],
    freeze_layer_norm: bool,
) -> nn.Module:
    """Freeze the layers of a huggingface clip text model."""
    if isinstance(freeze_layers, bool) and freeze_layers:
        for name, param in model.text_model.named_parameters():
            param.requires_grad = (
                (not freeze_layer_norm) if "LayerNorm" in name else False
            )

    layers = [  # NOTE: projection layer is not included
        model.text_model.embeddings,
        *model.text_model.encoder.layers,
        model.text_model.final_layer_norm,
    ]
    if isinstance(freeze_layers, float):
        freeze_layers = int(freeze_layers * len(layers))
    if isinstance(freeze_layers, int):
        freeze_layers = list(range(freeze_layers))

    if isinstance(freeze_layers, list):
        for idx, layer in enumerate(layers):
            if idx in freeze_layers:
                for name, param in layer.named_parameters():
                    param.requires_grad = (
                        (not freeze_layer_norm) if "LayerNorm" in name else False
                    )
    return model


def _freeze_vision_model(
    model: nn.Module,
    freeze_layers: Union[int, float, List[int], bool],
    freeze_layer_norm: bool,
) -> nn.Module:
    """Freeze the layers of a huggingface clip vision model."""
    if isinstance(freeze_layers, bool) and freeze_layers:
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = (
                (not freeze_layer_norm) if "LayerNorm" in name else False
            )

    layers = [  # NOTE: projection layer is not included
        model.vision_model.embeddings,
        model.vision_model.pre_layrnorm,
        *model.vision_model.encoder.layers,
        model.vision_model.post_layernorm,
    ]
    if isinstance(freeze_layers, float):
        freeze_layers = int(freeze_layers * len(layers))
    if isinstance(freeze_layers, int):
        freeze_layers = list(range(freeze_layers))

    if isinstance(freeze_layers, list):
        for idx, layer in enumerate(layers):
            if idx in freeze_layers:
                for name, param in layer.named_parameters():
                    param.requires_grad = (
                        (not freeze_layer_norm) if "LayerNorm" in name else False
                    )
    return model


def _warn_freeze_with_peft(
    peft_config: Optional["PeftConfig"], freeze_layers: Any
) -> None:
    """Raise a warning if both `peft_config` and `freeze_layers` are set."""
    if peft_config is not None and freeze_layers:
        rank_zero_warn(
            "Setting both `peft_config` and `freeze_layers` is not recommended. "
            "The `peft_config` will override the `freeze_layers` setting.",
            category=UserWarning,
        )
