"""Zero-shot classification evaluation task."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

import torch
from hydra_zen import store
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import move_data_to_device
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from mmlearn.datasets.core import CombinedDataset, Modalities
from mmlearn.datasets.core.data_collator import collate_example_list
from mmlearn.datasets.core.modalities import Modality
from mmlearn.tasks.hooks import EvaluationHooks


@dataclass
class ClassificationTaskSpec:
    """Specification for a classification task."""

    query_modality: str
    top_k: List[int]


@store(group="eval_task", provider="mmlearn")
class ZeroShotClassification(EvaluationHooks):
    """
    Zero-shot classification evaluation task.

    This task evaluates the zero-shot classification performance.

    Parameters
    ----------
    task_specs : List[ClassificationTaskSpec]
        A list of classification task specifications.
    """

    def __init__(
        self,
        task_specs: List[ClassificationTaskSpec],
        tokenizer: Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.task_specs = task_specs
        for spec in self.task_specs:
            assert Modalities.has_modality(spec.query_modality)

        self.metrics: Dict[tuple[Modality, int], MetricCollection] = {}
        self._embeddings_store: Dict[int, torch.Tensor] = {}

    def on_evaluation_epoch_start(self, pl_module: LightningModule) -> None:
        """Set up the evaluation task."""
        if pl_module.trainer.validating:
            eval_dataset: CombinedDataset = pl_module.trainer.val_dataloaders.dataset
        elif pl_module.trainer.testing:
            eval_dataset = pl_module.trainer.test_dataloaders.dataset
        else:
            raise ValueError(
                "ZeroShotClassification task is only supported for validation and testing."
            )

        self.all_dataset_info = {}

        # create metrics for each dataset/query_modality combination
        if not self.metrics:
            for dataset_index, dataset in enumerate(eval_dataset.datasets):
                try:
                    label_mapping = dataset.label_mapping
                except AttributeError:
                    raise ValueError(
                        "Dataset must have a `label_mapping` attribute to perform zero-shot classification."
                    ) from None

                try:
                    zero_shot_prompt_templates: list[str] = (
                        dataset.zero_shot_prompt_templates
                    )
                except AttributeError:
                    raise ValueError(
                        "Dataset must have a `zero_shot_prompt_templates` attribute to perform zero-shot classification."
                    ) from None

                num_classes = len(label_mapping)
                dataset_name = getattr(dataset, "name", dataset.__class__.__name__)

                self.all_dataset_info[dataset_index] = {
                    "name": dataset_name,
                    "label_mapping": label_mapping,
                    "prompt_templates": zero_shot_prompt_templates,
                    "num_classes": num_classes,
                }

                for spec in self.task_specs:
                    query_modality = Modalities.get_modality(spec.query_modality)
                    self.metrics[(query_modality, dataset_index)] = (
                        self._create_metrics(
                            num_classes,
                            spec.top_k,
                            prefix=f"{dataset_name}/{query_modality}_",
                            postfix="",
                        )
                    )

        for metric in self.metrics.values():
            metric.to(pl_module.device)

        # print(self.all_dataset_info.items())
        for dataset_index, dataset_info in self.all_dataset_info.items():
            label_mapping = dataset_info["label_mapping"]
            prompt_templates: list[str] = dataset_info["prompt_templates"]

            descriptions = [
                template.format(label)
                for label in label_mapping.values()
                for template in prompt_templates
            ]

            tokenized_descriptions = [
                self.tokenizer(description) for description in descriptions
            ]
            batch = move_data_to_device(
                collate_example_list(tokenized_descriptions),  # type: ignore
                pl_module.device,
            )

            with torch.no_grad():
                # TODO: encode in chunks for datasets with large number of classes
                # class_embeddings: torch.Tensor = pl_module.encode(
                #     batch, Modalities.TEXT
                # )
                chunk_size=64
                all_embeddings = []
                # Split the batch into chunks
                batch_size = len(next(iter(batch.values())))  # Get the length of the first item in the batch
                for i in range(0, batch_size, chunk_size):
                    # Get the current chunk of the batch (this handles the last chunk correctly)
                    batch_chunk = {k: v[i:min(i + chunk_size, batch_size)] for k, v in batch.items()}
                    
                    # Encode the chunk using the pl_module's encode method
                    chunk_embeddings = pl_module.encode(batch_chunk, Modalities.TEXT)
                    
                    # Append the chunk embeddings to the list
                    all_embeddings.append(chunk_embeddings)
                
                # Concatenate all chunk embeddings into a single tensor
                class_embeddings = torch.cat(all_embeddings, dim=0)
                
                
    
                print(f"class_embeddings 0 : {class_embeddings.shape}")
                print(f"prompt_templates 0 : {prompt_templates}")
                print(f"=============================== class_embeddings : {class_embeddings}")
                class_embeddings = class_embeddings.reshape(
                    len(label_mapping), len(prompt_templates), -1
                ).mean(dim=1)
                print(f"class_embeddings : {class_embeddings.shape}")
                class_embeddings /= class_embeddings.norm(dim=1, p=2,keepdim=True)
                print(f"class_embeddings 2 : {class_embeddings}")

            self._embeddings_store[dataset_index] = class_embeddings
            print("DONE")
    
    def evaluation_step(
        self,
        pl_module: LightningModule,
        batch: Dict[Union[str, Modality], torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Update metrics."""
        if pl_module.trainer.sanity_checking:
            return

        for (query_modality, dataset_index), metric_collection in self.metrics.items():
            # filter indices where dataset name is part of the metric_name
            batch_dataset_idx = batch["dataset_index"]

            # print(f"batch_dataset_idx : {batch_dataset_idx}")
            # print(f"dataset_index : {dataset_index}")
            # print(f"batch_dataset_idx == dataset_index : {batch_dataset_idx == dataset_index}")
            # get indices of examples from the dataset being evaluated
            matching_indices = torch.where(batch_dataset_idx == dataset_index)[0]

            if not matching_indices.numel():
                continue

            query_embeddings: torch.Tensor = pl_module.encode(batch, query_modality)
            # print(f"=============================== query_embeddings : {query_embeddings}")
            targets = batch[query_modality.target]  # True labels
            # print(f"========================== targets {targets}")

            query_embeddings = query_embeddings[matching_indices]
            targets = targets[matching_indices]
            # query_embeddings /= query_embeddings.norm(dim=1, p=2,keepdim=True)
            l2_norms = torch.norm(query_embeddings, p=2, dim=1)
            tolerance = 1e-6
            is_normalized = torch.allclose(l2_norms, torch.ones_like(l2_norms), atol=tolerance)

            # print(f"Is the tensor normalized? {is_normalized}")
            # print(f"*******************query_embeddings : {query_embeddings.shape}")

            # class_embeddings = torch.stack(
            #     [self._embeddings_store[dataset_index]] * len(query_embeddings)
            # )  # shape: (N_filtered, num_classes, embedding_dim)
            class_embeddings = self._embeddings_store[dataset_index]

            # query_embeddings = query_embeddings.unsqueeze(
            #     1
            # )  # shape: (N_filtered, 1, embedding_dim)
            # print(f"-------- query_embeddings : {query_embeddings.shape}")
            # print(f"-------- class_embeddings : {class_embeddings.shape}")

            if torch.float16 in (query_embeddings.dtype, class_embeddings.dtype):
                logits = (query_embeddings.float() @ class_embeddings.mT.float()).half()
            else:
                logits = query_embeddings @ class_embeddings.mT

            self.logits = logits.squeeze(1)  # shape: (N_filtered, num_classes)

            # print(f"---------------- logits : {logits}")
            metric_collection.update(logits, targets)

    def on_evaluation_epoch_end(self, pl_module: LightningModule) -> Dict[str, Any]:
        """Compute and reset metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """
        results = {}
        for metric_collection in self.metrics.values():
            results.update(metric_collection.compute())
            metric_collection.reset()

        self._embeddings_store.clear()
        print(results)
        return results

    @staticmethod
    def _create_metrics(
        num_classes: int, top_k: List[int], prefix: str, postfix: str
    ) -> MetricCollection:
        return MetricCollection(
            {
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "f1_score_macro": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "aucroc": AUROC(task="multiclass", num_classes=num_classes),
                **{
                    f"top{k}_accuracy": Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        top_k=k,
                        average="micro",
                    )
                    for k in top_k
                },
            },
            prefix=prefix,
            postfix=postfix,
        )
        