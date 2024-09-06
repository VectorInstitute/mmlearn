from hydra_zen import store
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors
from typing import Any, Literal
import torch
from torchmetrics.utilities.data import dim_zero_cat
from sklearn.metrics import f1_score as sklearn_f1_score

@store(group="modules/metrics", provider="mmlearn")
class ClassificationF1Score(Metric):
    """
    Classification F1 Score Metric.

    Computes the F1 score for classification tasks, specifically supporting scenarios such as zero-shot and linear probing where the model predicts on unseen data or after minimal tuning. The F1 score is a harmonic mean of precision and recall, useful for imbalanced datasets.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.
    mode : Literal["zero_shot", "linear_probing"]
        The mode of operation for classification, either 'zero_shot' or 'linear_probing'.
    top_k : int
        Specifies the top k predictions to consider for calculating metrics.
    reduction : {"mean", "sum", "none"}, default="mean"
        Specifies the method to reduce the F1 scores across batches.
    kwargs : Any
        Additional arguments to be passed to the torchmetrics.Metric base class.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, mode: Literal["zero_shot", "linear_probing"], top_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.mode = mode
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("true_indices", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, true_indices: torch.Tensor) -> None:
        """
        Accumulate predictions and true label indices for F1 score calculation.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities or embeddings, depending on the mode.
        true_indices : torch.Tensor
            True label indices for the predictions.
        """
        self.predictions.append(preds)
        self.true_indices.append(true_indices)

    def _is_distributed(self) -> bool:
        if self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        return distributed_available() if callable(distributed_available) else False

    def compute(self) -> torch.Tensor:
        """
        Compute the F1 score based on accumulated predictions and true indices.

        Returns
        -------
        torch.Tensor
            The computed F1 score based on the specified mode.
        """
        predictions = torch.cat(self.predictions, dim=0)
        true_indices = torch.cat(self.true_indices, dim=0)

        if self._is_distributed():
            predictions = torch.cat(gather_all_tensors(predictions), dim=0)
            true_indices = torch.cat(gather_all_tensors(true_indices), dim=0)

        if self.mode == "zero_shot":
            return self.zero_shot_f1(predictions, true_indices, self.target_embeddings)
        elif self.mode == "linear_probing":
            return self.linear_probing_f1(predictions, true_indices)

    def set_target_embeddings(self, target_embeddings: torch.Tensor) -> None:
        """
        Set the target embeddings for zero-shot classification.

        Parameters
        ----------
        target_embeddings : torch.Tensor
            Precomputed target embeddings for all labels, shape (num_classes, embedding_dim).
        """
        self.target_embeddings = target_embeddings

    def zero_shot_f1(self, predictions: torch.Tensor, true_indices: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute F1 score for zero-shot classification using cosine similarity.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted embeddings, shape (N, embedding_dim).
        true_indices : torch.Tensor
            True label indices, shape (N,).
        target_embeddings : torch.Tensor
            Precomputed target embeddings for all labels, shape (num_classes, embedding_dim).

        Returns
        -------
        torch.Tensor
            The computed F1 score for zero-shot classification.
        """
        # Compute cosine similarity between predictions and target embeddings
        similarities = torch.matmul(predictions, target_embeddings.t())  # Shape (N, num_classes)

        # Get the indices of the top-k most similar target embeddings for each prediction
        top_k_preds = similarities.topk(self.top_k, dim=1, largest=True, sorted=True)[1]  # Indices of top-k predictions

        # Convert top-k predictions into a binary matrix for precision and recall calculation
        pred_labels = torch.zeros_like(similarities).scatter_(1, top_k_preds, 1)
        true_labels = torch.zeros_like(similarities).scatter_(1, true_indices.view(-1, 1), 1)

        # Flatten the tensors to compute precision, recall, and F1
        pred_labels_flat = pred_labels.cpu().numpy().flatten()
        true_labels_flat = true_labels.cpu().numpy().flatten()

        # Calculate the F1 score using sklearn for multi-label classification
        f1 = sklearn_f1_score(true_labels_flat, pred_labels_flat, average='macro')

        return torch.tensor(f1)

    def linear_probing_f1(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute F1 score for linear probing classification using logits.

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
            The computed F1 score for linear probing classification.
        """
        # Get the indices of the top-k predictions
        top_k_preds = logits.topk(self.top_k, dim=1, largest=True, sorted=True)[1]  # Indices of top-k predictions

        # Convert top-k predictions into a binary matrix for precision and recall calculation
        pred_labels = torch.zeros_like(logits).scatter_(1, top_k_preds, 1)
        true_labels = torch.zeros_like(logits).scatter_(1, true_labels.view(-1, 1), 1)

        # Flatten the tensors to compute precision, recall, and F1
        pred_labels_flat = pred_labels.cpu().numpy().flatten()
        true_labels_flat = true_labels.cpu().numpy().flatten()

        # Calculate the F1 score using sklearn for multi-label classification
        f1 = sklearn_f1_score(true_labels_flat, pred_labels_flat, average='macro')

        return torch.tensor(f1)
