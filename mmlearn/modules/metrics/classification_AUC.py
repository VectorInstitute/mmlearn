from hydra_zen import store
from torchmetrics import Metric, AUROC
from torchmetrics.utilities.distributed import gather_all_tensors
from typing import Any, Callable, Literal, Optional, Union, List, Tuple
import torch
from sklearn.metrics import roc_auc_score

from torchmetrics.utilities.data import dim_zero_cat

@store(group="modules/metrics", provider="mmlearn")
class ClassificationAUC(Metric):
    """
    Zero-Shot Classification AUC metric.
    Computes the Area Under the ROC Curve (AUC) for zero-shot classification tasks where the model predicts
    labels that it has not seen during training.
    Parameters
    ----------
    reduction : {"mean", "sum", "none", None}, default="mean"
        Specifies the reduction to apply to the AUC scores.
    aggregation : {"mean", "median", "min", "max"} or callable, default="mean"
        Specifies the aggregation function to apply to the AUC scores computed
        in batches. If a callable is provided, it should accept a tensor of values
        and a keyword argument 'dim' and return a single scalar value.
    kwargs : Any
        Additional arguments to be passed to the torchmetrics.Metric class.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, top_k: int, mode: Literal["zero_shot", "linear_probing"], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.top_k = top_k

        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("true_indices", default=[], dist_reduce_fx=None)
        self.add_state("target_embeddings", default=[], dist_reduce_fx=None)


    def update(self, preds: torch.Tensor, true_indices: torch.Tensor, names) -> None:
        """
        Accumulate predictions and true label indices.
        Parameters
        ----------
        preds : torch.Tensor
            Predicted embeddings of shape `(N, embedding_dim)` where `N` is the number of samples.
        true_indices : torch.Tensor
            True label indices of shape `(N,)` where `N` is the number of samples.
        """
        self.predictions.append(preds)
        self.true_indices.append(true_indices)
        target_embeddings = [self.all_dataset_info[name].get_label_embedding() for name in names]
        self.target_embeddings.extend(target_embeddings)

    def _is_distributed(self) -> bool:
        if self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn
        return distributed_available() if callable(distributed_available) else False

    def set_all_dataset_info(self, all_dataset_info):
        self.all_dataset_info = all_dataset_info

    def compute(self) -> torch.Tensor:
        """
        Compute the AUC from the embeddings using dynamically provided target embeddings.
        Returns
        -------
        torch.Tensor
            Average of computed AUC scores.
        """
        predictions = torch.cat(self.predictions, dim=0)
        true_indices = torch.cat(self.true_indices, dim=0)
        target_embeddings = torch.stack(self.target_embeddings) if self.target_embeddings else None

        if self._is_distributed():
            predictions = torch.cat(gather_all_tensors(predictions), dim=0)
            true_indices = torch.cat(gather_all_tensors(true_indices), dim=0)
            if target_embeddings is not None:
                target_embeddings = torch.cat(gather_all_tensors(target_embeddings), dim=0)

        if self.mode == "zero_shot":
            return self.zero_shot_auc(predictions, true_indices, target_embeddings)
        elif self.mode == "linear_probing":
            return self.linear_probing_auc(predictions, true_indices)

    def zero_shot_auc(self, predictions: torch.Tensor, true_indices: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute AUC using cosine similarity for zero-shot classification.
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted embeddings, shape (N, embedding_dim).
        
        true_indices : torch.Tensor
            True label indices, shape (N,).
        
        target_embeddings : torch.Tensor
            Precomputed target embeddings for all labels per sample, shape (N, num_classes, embedding_dim).
        
        Returns
        -------
        torch.Tensor
            The AUC as a tensor.
        """

        N, num_classes, embedding_dim = target_embeddings.shape
        auc_scores = []

        for i in range(N):
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(predictions[i].unsqueeze(0), target_embeddings[i], dim=1)

            # Create labels array where true class is 1 and others are 0
            labels = torch.zeros(num_classes)
            labels[true_indices[i]] = 1

            # Calculate AUC for this sample
            if torch.sum(labels) > 0:  # Ensure there is at least one positive class
                auc = roc_auc_score(labels.detach().cpu().numpy(), cos_sim.detach().cpu().numpy())
                auc_scores.append(auc)

        # Calculate mean AUC across all samples
        mean_auc = torch.tensor(auc_scores).mean().to(predictions.device)
        return mean_auc

    def linear_probing_auc(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute AUC for linear probing classification using logits.
        Parameters
        ----------
        logits : torch.Tensor
            Logits output by the linear classifier, shape (N, num_classes),
            where N is the number of examples, and num_classes is the number of classes.
        
        true_labels : torch.Tensor
            Ground truth label indices, shape (N,).
            
        Returns
        -------
        torch.Tensor
            The AUC as a tensor.
        """
        N, num_classes = logits.shape
        softmax_probs = torch.nn.functional.softmax(logits, dim=1)
        auc_scores = []

        for i in range(N):
            # Create labels array for each sample
            labels = torch.zeros(num_classes)
            labels[true_labels[i]] = 1

            # Extract the softmax probabilities for each class
            probs = softmax_probs[i]

            if torch.sum(labels) > 0:  # Ensure there is at least one positive class
                auc = roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())
                auc_scores.append(auc)

        mean_auc = torch.tensor(auc_scores).mean().to(logits.device)
        return mean_auc