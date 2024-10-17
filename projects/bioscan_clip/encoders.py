import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftConfig
from torch import nn
from transformers import BertConfig, BertForMaskedLM
from transformers.modeling_outputs import BaseModelOutput

from mmlearn import hf_utils
from mmlearn.datasets.core.modalities import Modalities


class BarcodeBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        projection_dim: int,
        pretrained_checkpoint_path: Optional[str] = None,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        peft_config: Optional[PeftConfig] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            output_hidden_states=True,
            **model_config_kwargs or {},
        )
        model = BertForMaskedLM(config)

        if pretrained_checkpoint_path is not None:
            state_dict = torch.load(
                pretrained_checkpoint_path, map_location=torch.device("cpu")
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            if len(incompatible_keys.missing_keys) > 0:
                warnings.warn(
                    f"Missing keys in the pretrained checkpoint: {incompatible_keys.missing_keys}",
                    stacklevel=1,
                )
            if len(incompatible_keys.unexpected_keys) > 0:
                warnings.warn(
                    f"Unexpected keys in the pretrained checkpoint: {incompatible_keys.unexpected_keys}",
                    stacklevel=1,
                )

        if isinstance(freeze_layers, bool) and freeze_layers:
            for name, param in model.named_parameters():
                param.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in name else False
                )

        modules = [model.bert.embeddings, *model.bert.encoder.layer]
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

        model.cls.predictions.decoder = nn.Linear(
            model.cls.predictions.decoder.in_features, projection_dim
        )
        self.model = model

        if peft_config is not None:
            self.model = hf_utils._wrap_peft_model(self.model, peft_config)

    def forward(self, inputs: Dict[str, Any]) -> BaseModelOutput:
        """Run the forward pass."""
        outputs = self.model(
            input_ids=inputs[Modalities.DNA.name],
            attention_mask=inputs.get(
                "attention_mask", inputs.get(Modalities.DNA.attention_mask, None)
            ),
            position_ids=inputs.get("position_ids"),
            output_attentions=inputs.get("output_attentions"),
            return_dict=True,
        )

        return BaseModelOutput(
            last_hidden_state=outputs.logits.softmax(dim=-1).mean(dim=1),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
