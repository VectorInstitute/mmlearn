from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from hydra_zen import store

import torch
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Metric, MetricCollection

import mmlearn
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core.data_collator import collate_example_list
from mmlearn.modules.metrics.classification_accuracy import ClassificationAccuracy
from mmlearn.tasks.hooks import EvaluationHooks
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from typing import Callable, Optional, Literal
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from mmlearn.constants import NAME_KEY


class Mode(str, Enum):
    ZERO_SHOT = "zero_shot"
    LINEAR_PROBING = "linear_probing"

@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""
    metric_name: str
    query_modality: str
    top_k: List[int]
    mode: Mode

@store(group="eval_task", provider="mmlearn")
class Classification(EvaluationHooks):
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
        self.metrics: Dict[str, Metric] = {}

        self.metrics = {}

        for spec in self.task_specs:
            assert Modalities.has_modality(spec.query_modality)
            query_modality = Modalities.get_modality(spec.query_modality)
            mode = Mode(spec.mode)
            metric_name = spec.metric_name
            metric_class = getattr(mmlearn.modules.metrics, metric_name)
            self.metrics.update({
                (query_modality, mode, metric_name): MetricCollection(
                    {
                        f"{query_modality}_{mode}_C_{metric_name}@{k}": metric_class(
                            top_k=k,
                            mode=Mode(spec.mode)
                        )
                        for k in spec.top_k
                    }
                )
            })


        self.tokenizer = tokenizer


    def on_evaluation_epoch_start(self, pl_module: LightningModule, all_dataset_info) -> None:
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

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set in the dataset to generate tokenized label descriptions")

        # Set the label embeddings
        self.target_embeddings = {}
        for name, dataset_info in all_dataset_info.items():
            descriptions = ["This image has a sign of " + label for label in dataset_info.get_label_mapping().values()]

            dataset = LabelDescriptionDataset(descriptions, self.tokenizer)
            batch_size = len(dataset)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_example_list)
            batch = next(iter(dataloader))
            batch = {key: value.to(pl_module.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            all_dataset_info[name].set_label_embedding(pl_module(batch)[Modalities.get_modality(Modalities.TEXT).embedding])

        for metric_collection in self.metrics.values():
            for metric_name, metric in metric_collection.items():
                if hasattr(metric, 'set_all_dataset_info'):
                    metric.set_all_dataset_info(all_dataset_info)

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

        for (query_modality, mode, metric_name), metric in self.metrics.items():
            output_embeddings = outputs[query_modality.embedding] # Input image embedding
            label_index = batch[query_modality.target] # True label index
            names = batch[NAME_KEY]

            metric.update(output_embeddings, label_index, names)

    def on_evaluation_epoch_end(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Compute the classification accuracy metrics.
        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        results = {}
        for (query_modality, mode, metric_name), metric in self.metrics.items():
            results.update(metric.compute())
            metric.reset()
        return results
    