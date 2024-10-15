import logging
import sys
from typing import Any, Dict, Optional, Union

import faiss
import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import move_data_to_device
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import normalize

from mmlearn.conf import external_store
from mmlearn.datasets.core import find_matching_indices
from mmlearn.datasets.core.modalities import Modalities, Modality
from mmlearn.tasks.hooks import EvaluationHooks


logger = logging.getLogger(__name__)

All_TYPE_OF_FEATURES_OF_QUERY = [
    Modalities.RGB.embedding,
    Modalities.DNA.embedding,
    Modalities.TEXT.embedding,
    "averaged_embedding",
    "concatenated_embedding",
]
All_TYPE_OF_FEATURES_OF_KEY = [
    Modalities.RGB.embedding,
    Modalities.DNA.embedding,
    Modalities.TEXT.embedding,
    "averaged_embedding",
    "concatenated_embedding",
    "all_key_embedding",
]
LEVELS = ["order", "family", "genus", "species"]


@external_store(group="eval_task", top_k=[1])
class TaxonomicClassification(EvaluationHooks):
    def __init__(self, top_k: list[int]):
        super().__init__()
        self.top_k = top_k

    def on_evaluation_epoch_start(self, pl_module: pl.LightningModule) -> None:
        # initialize the dictionary to store the embeddings and labels
        self._embedding_store: Dict[str, Dict[str, Any]] = {}

    def evaluation_step(  # noqa: PLR0912
        self, pl_module: pl.LightningModule, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        if pl_module.trainer.sanity_checking:
            return

        assert (
            Modalities.RGB.name in batch
            and Modalities.DNA.name in batch
            and Modalities.TEXT.name in batch
        ), "The batch must contain the RGB, DNA and text modalities"

        outputs: Dict[str, Any] = pl_module(batch)

        splits_batch = batch["split"]
        labels_batch = batch["labels"]
        process_ids_batch = batch["process_id"]

        if pl_module.trainer._accelerator_connector.is_distributed:
            batch = pl_module.all_gather(batch)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = torch.cat(torch.unbind(batch[key]))
                elif isinstance(batch[key], dict):
                    for k in batch[key]:
                        if isinstance(batch[key][k], torch.Tensor):
                            batch[key][k] = torch.cat(torch.unbind(batch[key][k]))

            outputs = pl_module.all_gather(outputs)
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = torch.cat(torch.unbind(outputs[key]))
                elif isinstance(outputs[key], dict):
                    for k in outputs[key]:
                        if isinstance(outputs[key][k], torch.Tensor):
                            outputs[key][k] = torch.cat(torch.unbind(outputs[key][k]))

            splits_batch = _gather_obj(splits_batch)
            process_ids_batch = _gather_obj(process_ids_batch)

            for key in labels_batch:
                labels_batch[key] = _gather_obj(labels_batch[key])

        # move tensors to CPU
        batch = move_data_to_device(batch, "cpu")
        outputs = move_data_to_device(outputs, "cpu")

        _, rgb_indices = find_matching_indices(
            batch["example_ids"]["split"], batch["example_ids"][Modalities.RGB.name]
        )
        _, dna_indices = find_matching_indices(
            batch["example_ids"]["split"], batch["example_ids"][Modalities.DNA.name]
        )
        _, text_indices = find_matching_indices(
            batch["example_ids"]["split"], batch["example_ids"][Modalities.TEXT.name]
        )

        splits = set(splits_batch)
        for split in splits:
            if split not in self._embedding_store:
                self._embedding_store[split] = {}

            # get the indices of the examples in the current split
            split_indices = torch.asarray(
                np.where(np.asarray(splits_batch) == split)[0],
            )

            # get the embeddings for the current split
            rgb_embeddings = outputs[Modalities.RGB.embedding][
                torch.where(torch.isin(rgb_indices, split_indices))[0]
            ]
            dna_embeddings = outputs[Modalities.DNA.embedding][
                torch.where(torch.isin(dna_indices, split_indices))[0]
            ]
            text_embeddings = outputs[Modalities.TEXT.embedding][
                torch.where(torch.isin(text_indices, split_indices))[0]
            ]

            process_ids = np.asarray(process_ids_batch)[split_indices].tolist()
            labels = np.asarray(_convert_label_dict_to_list_of_dict(labels_batch))[
                split_indices
            ]
            labels = labels.tolist() if isinstance(labels, np.ndarray) else [labels]

            # concatenate the embeddings
            concatenated_embeddings = torch.cat([rgb_embeddings, dna_embeddings], dim=1)
            averaged_embeddings = torch.mean(
                torch.stack([rgb_embeddings, dna_embeddings]), dim=0
            )

            # store the embeddings
            self._embedding_store[split].setdefault(
                Modalities.RGB.embedding, []
            ).append(rgb_embeddings)
            self._embedding_store[split].setdefault(
                Modalities.DNA.embedding, []
            ).append(dna_embeddings)
            self._embedding_store[split].setdefault(
                Modalities.TEXT.embedding, []
            ).append(text_embeddings)
            self._embedding_store[split].setdefault(
                "concatenated_embedding", []
            ).append(concatenated_embeddings)
            self._embedding_store[split].setdefault("averaged_embedding", []).append(
                averaged_embeddings
            )
            self._embedding_store[split].setdefault("process_ids", []).extend(
                process_ids
            )
            self._embedding_store[split].setdefault("labels", []).extend(labels)

            if split == "all_keys":
                self._embedding_store[split].setdefault("all_key_embedding", []).append(
                    torch.cat([rgb_embeddings, dna_embeddings, text_embeddings], dim=0)
                )
                self._embedding_store[split].setdefault("all_key_labels", []).extend(
                    labels + labels + labels
                )

    def on_evaluation_epoch_end(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        # concatenate the embeddings for the entire dataset
        if not self._embedding_store:
            return {}

        for split in self._embedding_store:
            self._embedding_store[split] = {
                key: torch.cat(value, dim=0)
                if isinstance(value[0], torch.Tensor)
                else value
                for key, value in self._embedding_store[split].items()
            }

        acc_dict, _, _ = _inference_and_print_result(
            self._embedding_store["all_keys"],
            self._embedding_store["val_seen"],
            self._embedding_store["val_unseen"],
            k_list=self.top_k,
        )
        _print_micro_and_macro_acc(acc_dict, self.top_k)

        # flatten the accuracy dictionary
        rgb_2_dna_res = acc_dict[Modalities.RGB.embedding][Modalities.DNA.embedding]
        results = {
            f"{split}_rgb_2_dna_top_{k}_{type_of_acc}_{level}": value
            for split in rgb_2_dna_res
            for type_of_acc in rgb_2_dna_res[split]
            for k in rgb_2_dna_res[split][type_of_acc]
            for level, value in rgb_2_dna_res[split][type_of_acc][k].items()
        }

        # clear the embeddings store
        self._embedding_store.clear()

        return results


def _convert_label_dict_to_list_of_dict(
    label_batch: Dict[str, np.str_],
) -> list[Dict[str, np.str_]]:
    order = label_batch["order"]

    family = label_batch["family"]
    genus = label_batch["genus"]
    species = label_batch["species"]

    return [
        {"order": o, "family": f, "genus": g, "species": s}
        for o, f, g, s in zip(order, family, genus, species)
    ]


def _gather_obj(local_objs: Any) -> list[Any]:
    world_size = dist.get_world_size()
    gathered_objs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_objs, local_objs)

    # flatten the list of lists into a single list
    return [item for obj_list in gathered_objs for item in obj_list]


def _make_prediction(
    query_feature: torch.Tensor,
    keys_feature: torch.Tensor,
    keys_label: list[dict[str, str]],
    with_similarity: bool = False,
    with_indices: bool = False,
    max_k: int = 5,
) -> Union[list, list[list]]:
    index = faiss.IndexFlatIP(keys_feature.shape[-1])
    keys_feature = normalize(keys_feature, norm="l2", axis=1).astype(np.float32)
    query_feature = normalize(query_feature, norm="l2", axis=1).astype(np.float32)
    index.add(keys_feature)
    pred_list = []

    similarities, indices = index.search(query_feature, max_k)
    for key_indices in indices:
        k_pred_in_diff_level: dict[str, list[str]] = {}
        for level in LEVELS:
            if level not in k_pred_in_diff_level:
                k_pred_in_diff_level[level] = []
            for i in key_indices:
                try:
                    k_pred_in_diff_level[level].append(keys_label[i][level])
                except Exception as e:
                    print(keys_label + "\n" + str(i) + "\n" + str(e))
                    sys.exit()
        pred_list.append(k_pred_in_diff_level)

    out = [pred_list]

    if with_similarity:
        out.append(similarities)

    if with_indices:
        out.append(indices)

    if len(out) == 1:
        return out[0]
    return out


def _top_k_micro_accuracy(
    pred_list: Union[list, list[list]],
    gt_list: dict[str, str],
    k_list: Optional[list[int]] = None,
) -> dict[int, dict[str, float]]:
    total_samples = len(pred_list)
    k_micro_acc = {}
    for k in k_list:
        if k not in k_micro_acc:
            k_micro_acc[k] = {}
        for level in LEVELS:
            correct_in_curr_level = 0
            for pred_dict, gt_dict in zip(pred_list, gt_list):
                pred_labels = pred_dict[level][:k]
                gt_label = gt_dict[level]
                if gt_label in pred_labels:
                    correct_in_curr_level += 1
            k_micro_acc[k][level] = correct_in_curr_level * 1.0 / total_samples

    return k_micro_acc


def _top_k_macro_accuracy(
    pred_list: Union[list, list[list]],
    gt_list: dict[str, str],
    k_list: Optional[list[int]] = None,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, dict[str, float]]]]:
    if k_list is None:
        k_list = [1, 3, 5]

    macro_acc_dict = {}
    per_class_acc = {}
    pred_counts = {}
    gt_counts = {}

    for k in k_list:
        macro_acc_dict[k] = {}
        per_class_acc[k] = {}
        pred_counts[k] = {}
        gt_counts[k] = {}
        for level in LEVELS:
            pred_counts[k][level] = {}
            gt_counts[k][level] = {}
            for pred, gt in zip(pred_list, gt_list):
                pred_labels = pred[level][:k]
                gt_label = gt[level]
                if gt_label not in pred_counts[k][level]:
                    pred_counts[k][level][gt_label] = 0
                if gt_label not in gt_counts[k][level]:
                    gt_counts[k][level][gt_label] = 0

                if gt_label in pred_labels:
                    pred_counts[k][level][gt_label] = (
                        pred_counts[k][level][gt_label] + 1
                    )
                gt_counts[k][level][gt_label] = gt_counts[k][level][gt_label] + 1

    for k in k_list:
        for level in LEVELS:
            sum_in_this_level = 0
            list_of_labels = list(gt_counts[k][level].keys())
            per_class_acc[k][level] = {}
            for gt_label in list_of_labels:
                sum_in_this_level = (
                    sum_in_this_level
                    + pred_counts[k][level][gt_label]
                    * 1.0
                    / gt_counts[k][level][gt_label]
                )
                per_class_acc[k][level][gt_label] = (
                    pred_counts[k][level][gt_label]
                    * 1.0
                    / gt_counts[k][level][gt_label]
                )
            macro_acc_dict[k][level] = sum_in_this_level / len(list_of_labels)

    return macro_acc_dict, per_class_acc


def _print_micro_and_macro_acc(
    acc_dict: dict[str, dict[str, dict[str, dict[str, float]]]], k_list: list[int]
) -> None:
    console = Console()
    table = Table(expand=True)

    header = [
        "Seen Order",
        "Seen Family",
        "Seen Genus",
        "Seen Species",
        "Unseen Order",
        "Unseen Family",
        "Unseen Genus",
        "Unseen Species",
    ]
    table.add_column(" ", width=70)
    for header_name in header:
        table.add_column(header_name, justify="center")

    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        if query_feature_type not in acc_dict:
            continue

        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            if key_feature_type not in acc_dict[query_feature_type]:
                continue

            for type_of_acc in ["micro_acc", "macro_acc"]:
                for k in k_list:
                    if (
                        len(list(acc_dict[query_feature_type][key_feature_type].keys()))
                        == 0
                    ):
                        continue
                    curr_row = [
                        f"Query_feature: {query_feature_type} || Key_feature: {key_feature_type} || {type_of_acc} top_{k}"
                    ]

                    for split in ["seen", "unseen"]:
                        for level in LEVELS:
                            num = round(
                                acc_dict[query_feature_type][key_feature_type][split][
                                    type_of_acc
                                ][k][level],
                                4,
                            )

                            curr_row.append(f"\t{num}")

                    table.add_row(*curr_row)

    console.print(table, highlight=True, crop=False, new_line_start=True)


def _inference_and_print_result(
    keys_dict: Dict[str, Any],
    seen_dict: Dict[str, Any],
    unseen_dict: Dict[str, Any],
    k_list: Optional[list[int]] = None,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]]],
    dict[str, dict[str, dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]]],
    dict[str, dict[str, dict[str, dict[str, Union[list, list[list]]]]]],
]:
    if k_list is None:
        k_list = [1, 3, 5]
    max_k = k_list[-1]

    seen_gt_label = seen_dict["labels"]
    unseen_gt_label = unseen_dict["labels"]
    keys_label = keys_dict["labels"]

    acc_dict: dict[
        str, dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]]
    ] = {}
    per_class_acc: dict[
        str, dict[str, dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]]]
    ] = {}
    pred_dict: dict[str, dict[str, dict[str, dict[str, Union[list, list[list]]]]]] = {}

    for query_feature_type in All_TYPE_OF_FEATURES_OF_QUERY:
        if query_feature_type not in seen_dict:
            continue

        acc_dict[query_feature_type] = {}
        per_class_acc[query_feature_type] = {}
        pred_dict[query_feature_type] = {}

        for key_feature_type in All_TYPE_OF_FEATURES_OF_KEY:
            if key_feature_type not in keys_dict:
                continue

            acc_dict[query_feature_type][key_feature_type] = {}
            per_class_acc[query_feature_type][key_feature_type] = {}
            pred_dict[query_feature_type][key_feature_type] = {}

            curr_seen_feature: Optional[torch.Tensor] = seen_dict[query_feature_type]
            curr_unseen_feature: Optional[torch.Tensor] = unseen_dict[
                query_feature_type
            ]

            curr_keys_feature: Optional[torch.Tensor] = keys_dict[key_feature_type]
            if curr_keys_feature is None:
                continue
            if key_feature_type == "all_key_embedding":
                keys_label = keys_dict["all_key_labels"]

            if (
                curr_keys_feature is None
                or curr_seen_feature is None
                or curr_unseen_feature is None
                or curr_keys_feature.shape[-1] != curr_seen_feature.shape[-1]
                or curr_keys_feature.shape[-1] != curr_unseen_feature.shape[-1]
            ):
                continue

            curr_seen_pred_list = _make_prediction(
                curr_seen_feature,
                curr_keys_feature,
                keys_label,
                with_similarity=False,
                max_k=max_k,
            )
            curr_unseen_pred_list = _make_prediction(
                curr_unseen_feature, curr_keys_feature, keys_label, max_k=max_k
            )

            pred_dict[query_feature_type][key_feature_type] = {
                "curr_seen_pred_list": curr_seen_pred_list,
                "curr_unseen_pred_list": curr_unseen_pred_list,
            }

            logger.info(
                "Computing accuracy for query feature type: %s and key feature type: %s",
                query_feature_type,
                key_feature_type,
            )
            acc_dict[query_feature_type][key_feature_type]["seen"] = {}
            acc_dict[query_feature_type][key_feature_type]["unseen"] = {}
            acc_dict[query_feature_type][key_feature_type]["seen"]["micro_acc"] = (
                _top_k_micro_accuracy(curr_seen_pred_list, seen_gt_label, k_list=k_list)
            )
            acc_dict[query_feature_type][key_feature_type]["unseen"]["micro_acc"] = (
                _top_k_micro_accuracy(
                    curr_unseen_pred_list, unseen_gt_label, k_list=k_list
                )
            )

            seen_macro_acc, seen_per_class_acc = _top_k_macro_accuracy(
                curr_seen_pred_list, seen_gt_label, k_list=k_list
            )

            unseen_macro_acc, unseen_per_class_acc = _top_k_macro_accuracy(
                curr_unseen_pred_list, unseen_gt_label, k_list=k_list
            )

            per_class_acc[query_feature_type][key_feature_type]["seen"] = (
                seen_per_class_acc
            )
            per_class_acc[query_feature_type][key_feature_type]["unseen"] = (
                unseen_per_class_acc
            )

            acc_dict[query_feature_type][key_feature_type]["seen"]["macro_acc"] = (
                seen_macro_acc
            )
            acc_dict[query_feature_type][key_feature_type]["unseen"]["macro_acc"] = (
                unseen_macro_acc
            )

    return acc_dict, per_class_acc, pred_dict
