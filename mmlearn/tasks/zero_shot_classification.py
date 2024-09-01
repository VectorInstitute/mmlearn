from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple, Union
from hydra_zen import store

import torch
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import MetricCollection

from mmlearn.datasets.core.example import Example, collate_example_list
from mmlearn.modules.metrics.classification_accuracy import ZeroShotClassificationAccuracy
from mmlearn.tasks.hooks import EvaluationHooks
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader

@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""
    top_k: List[int]

@store(group="eval_task", provider="mmlearn")
class ZeroShotClassification(EvaluationHooks):
    """Zero-shot classification evaluation task.

    This task evaluates the zero-shot classification performance of a model on a dataset.

    Parameters
    ----------
    task_specs : List[ClassificationTaskSpec]
        A list of classification task specifications, each defining the number of classes
        and the top-k values for accuracy measurement.
    """
    
    def __init__(self, task_specs: List[ClassificationTaskSpec],
                 tokenizer: Optional[Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]] = None,):
        super().__init__()
        self.task_specs = task_specs
        self.metrics: Dict[str, ZeroShotClassificationAccuracy] = {}

        for spec in self.task_specs:
            self.metrics[f"ZSC_Accuracy_Top{spec.top_k}"] = ZeroShotClassificationAccuracy(
                top_k=tuple(spec.top_k))
            
        self.tokenizer = tokenizer
        # TODO - This should go in dataloader
        label_mapping = {
            "nv": "melanocytic nevus",
            "mel": "melanoma",
            "bkl": "benign keratosis",
            "bcc": "basal cell carcinoma",
            "akiec": "actinic keratosis",
            "vasc": "vascular lesion",
            "df": "dermatofibroma"
        }
        
        if tokenizer is None:
            raise ValueError("Tokenizer must be set in the dataset to generate tokenized label descriptions")

        # Create descriptive text for each label
        self.descriptions = ["This image has a sign of " + label for label in label_mapping.values()]
            
    def on_evaluation_epoch_start(self, pl_module: LightningModule) -> None:
        """Move the metrics to the device of the Lightning module."""
        for metric in self.metrics.values():
            metric.to(pl_module.device)
            
            
        class LabelDescriptionDataset(Dataset):
            def __init__(self, descriptions, tokenizer):
                self.descriptions = descriptions
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.descriptions)

            def __getitem__(self, idx):
                description = self.descriptions[idx]
                print(f"++++++++++++++ description : {description}")
                tokens = self.tokenizer(description)
                
                example = Example(
                    {
                        Modalities.RGB: torch.rand(3, 224, 224),
                        Modalities.TEXT: description,
                    }
                )
                
                if tokens is not None:
                    if isinstance(tokens, dict):  # output of HFTokenizer
                        assert (
                            Modalities.TEXT in tokens
                        ), f"Missing key `{Modalities.TEXT}` in tokens."
                        example.update(tokens)
                    else:
                        example[Modalities.TEXT] = tokens
                        
                return example
            
        dataset = LabelDescriptionDataset(self.descriptions, self.tokenizer)
        batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_example_list)
        # for batch in dataloader:
        batch = next(iter(dataloader))
        batch = {key: value.to(pl_module.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            # print("!!!!!!!!!!!!!!!!")
        self.target_embedding: Dict[Union[str, Modality], Any] = pl_module(batch)[Modalities.get_modality(Modalities.TEXT).embedding]
        
        logging.info(f"-- target_embedding shape : {self.target_embedding.shape}")
        # logging.info(f"-- target_embedding shape : {self.target_embedding.keys()}")
    
    def evaluation_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Update classification accuracy metrics."""
        if trainer.sanity_checking:
            return
        
        outputs: Dict[Union[str, Modality], Any] = pl_module(batch)
        
        # TODO add for loop here
        output_embeddings = outputs[Modalities.get_modality(Modalities.RGB).embedding] # Input image embedding
        label_index = batch[Modalities.get_modality(Modalities.RGB).target] # True label index
        
        
        for name, metric in self.metrics.items():
            metric.update(output_embeddings, label_index)
    
    def on_evaluation_epoch_end(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Compute the classification accuracy metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute(self.target_embedding)
            metric.reset()
        return results
    