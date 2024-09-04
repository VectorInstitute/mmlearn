import os

from omegaconf import MISSING
from hydra_zen import builds
from peft import LoraConfig
import torch

from mmlearn.conf import external_store
from mmlearn.modules.encoders.hf_text_encoders import HFTextEncoder

from projects.bioscan_clip.encoders import TimmViT, BarcodeBERT
from projects.bioscan_clip.dataset import BIOSCANInsectDataset
from projects.bioscan_clip.eval_task import TaxonomicClassification
from projects.bioscan_clip.dna_tokenizer import DNAProcessor


# configurations for encoders
class MeanPooler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


external_store(
    HFTextEncoder,
    name="bert-small-lora",
    group="modules/encoders",
    model_name_or_path="prajjwal1/bert-small",
    pooling_layer=builds(MeanPooler),
    peft_config=builds(
        LoraConfig, populate_full_signature=True, r=4, target_modules=["query", "value"]
    ),
    hydra_convert="object",  # required for `peft_config` to be converted to a `PeftConfig` object
)
external_store(
    TimmViT,
    name="timm-vit-lora",
    group="modules/encoders",
    model_name="vit_base_patch16_224",
    peft_config=builds(
        LoraConfig,
        populate_full_signature=True,
        r=4,
        modules_to_save=["head"],  # don't freeze the projection head
        target_modules=["qkv"],
    ),
    hydra_convert="object",
)
external_store(
    BarcodeBERT,
    name="barcode-bert-lora",
    group="modules/encoders",
    pretrained_checkpoint_path=os.getenv("BARCODEBERT_5MER"),
    vocab_size=1027,
    projection_dim=768,
    peft_config=builds(
        LoraConfig,
        populate_full_signature=True,
        r=4,
        target_modules=["query", "value"],
        modules_to_save=["decoder"],
    ),
    hydra_convert="object",
)

# dataset configuration
external_store(
    BIOSCANInsectDataset,
    name="BIOSCAN-1M",
    group="datasets",
    variant="1m",
    dna_processor=builds(DNAProcessor, populate_full_signature=True, max_length=660),
    path_to_hdf5_file=os.getenv("BIOSCAN_1M_HDF5", MISSING),
    split=MISSING,
)

external_store(
    BIOSCANInsectDataset,
    name="BIOSCAN-5M",
    group="datasets",
    variant="5m",
    dna_processor=builds(DNAProcessor, populate_full_signature=True, max_length=660),
    path_to_hdf5_file=os.getenv("BIOSCAN_5M_HDF5", MISSING),
    split=MISSING,
)
