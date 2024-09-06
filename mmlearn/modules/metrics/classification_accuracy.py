from hydra_zen import store
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors
from typing import Any, Callable, Literal, Optional, Union, List, Tuple
import torch

from torchmetrics.utilities.data import dim_zero_cat

@store(group="modules/metrics", provider="mmlearn")
class ClassificationAccuracy(Metric):
    """
    Zero-Shot Classification Accuracy metric.

    Computes accuracy for zero-shot classification tasks where the model predicts
    labels that it has not seen during training.

    Parameters
    ----------
    num_classes : int
        The number of classes to consider in the zero-shot classification task.
    reduction : {"mean", "sum", "none", None}, default="mean"
        Specifies the reduction to apply to the accuracy scores.
    aggregation : {"mean", "median", "min", "max"} or callable, default="mean"
        Specifies the aggregation function to apply to the accuracy scores computed
        in batches. If a callable is provided, it should accept a tensor of values
        and a keyword argument 'dim' and return a single scalar value.
    kwargs : Any
        Additional arguments to be passed to the torchmetrics.Metric class.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        mode: Literal["zero_shot", "linear_probing"],
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.mode = mode

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
        target_embeddings = torch.tensor([])
        target_embeddings = [self.all_target_embeddings[name] for name in names]

        # Convert the list of embeddings to a tensor of shape (N, embedding_dim)
        self.target_embeddings = self.target_embeddings + target_embeddings
        
    def _is_distributed(self) -> bool:
        if self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        return distributed_available() if callable(distributed_available) else False  # type: ignore[no-any-return]

    def set_all_target_embeddings(self, all_target_embeddings):
        self.all_target_embeddings = all_target_embeddings
    
    def compute(self) -> torch.Tensor:
        """
        Compute the top-k accuracies from the embeddings using dynamically provided target embeddings.

        Parameters
        ----------
        target_embeddings : torch.Tensor
            Target embeddings for all labels (shape `[num_classes, embedding_dim]`).

        Returns
        -------
        torch.Tensor
            Average of computed top-k accuracies.
        """
        predictions = torch.cat(self.predictions, dim=0)
        true_indices = torch.cat(self.true_indices, dim=0)
        target_embeddings = torch.stack(self.target_embeddings)

        if self._is_distributed():
            predictions = torch.cat(gather_all_tensors(predictions), dim=0)
            true_indices = torch.cat(gather_all_tensors(true_indices), dim=0)
            target_embeddings = torch.cat(gather_all_tensors(target_embeddings), dim=0)
            
        if self.mode == "zero_shot":
            # make target_embeddings a self method
            return self.zero_shot_accuracy(predictions, true_indices, target_embeddings)
        elif self.mode == "linear_probing":
            return self.linear_probing_accuracy(predictions, true_indices)
    
    
    def zero_shot_accuracy(self, predictions: torch.Tensor, true_indices: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute top-k zero-shot accuracy using cosine similarity.

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
            The top-k zero-shot accuracy as a tensor.
        """

        # Initialize list to store similarities for each sample
        similarities = []
        
        # Loop over each sample to compute cosine similarity individually
        for i in range(predictions.shape[0]):
            # Compute cosine similarity for the i-th sample: shape (1, num_classes)
            sim = torch.matmul(predictions[i].view(1, -1), target_embeddings[i].transpose(0, 1))
            similarities.append(sim)
        
        # Concatenate all similarity results: shape (N, num_classes)
        similarities = torch.cat(similarities, dim=0)

        # Get the indices of the top-k most similar target embeddings for each prediction
        top_k_preds = similarities.topk(self.top_k, dim=1, largest=True, sorted=True)[1]  # Indices of top-k predictions
        
        # Compare the top-k predictions with the true indices
        correct = top_k_preds.eq(true_indices.view(-1, 1).expand_as(top_k_preds))
        
        # Calculate the number of correct predictions in the top-k
        correct_k = correct[:, :self.top_k].reshape(-1).float().sum(0, keepdim=True)
        
        # Compute the accuracy as the ratio of correct predictions
        accuracy = correct_k / true_indices.size(0)
        
        return accuracy

    
    def linear_probing_accuracy(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute top-k accuracy for linear probing classification using logits.

        Parameters
        ----------
        logits : torch.Tensor
            Logits output by the linear classifier, shape (N, num_classes), 
            where N is the number of examples, and num_classes is the number of classes.
        
        true_labels : torch.Tensor
            Ground truth label indices, shape (N,).
        
        k : int
            The value of k for which to compute the top-k accuracy.
        
        Returns
        -------
        torch.Tensor
            The top-k accuracy as a tensor.
        """
        # Get the indices of the top-k predictions
        top_k_preds = logits.topk(self.top_k, dim=1, largest=True, sorted=True)[1]  # Indices of top-k predictions

        # Compare the top-k predictions with the true labels
        correct = top_k_preds.eq(true_labels.view(-1, 1).expand_as(top_k_preds))
        
        # Calculate the number of correct predictions in the top-k
        correct_k = correct[:, :self.top_k].reshape(-1).float().sum(0, keepdim=True)
        
        # Compute the accuracy as the ratio of correct predictions
        accuracy = correct_k / true_labels.size(0)
        
        return accuracy
    