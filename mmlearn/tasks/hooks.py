"""Task-related hooks for Lightning modules."""

from typing import Any, Mapping, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class EvaluationHooks:
    """Hooks for evaluation."""

    def on_evaluation_epoch_start(self, pl_module: pl.LightningModule) -> None:
        """Prepare the evaluation loop.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        """

    def evaluation_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> Optional[Mapping[str, Any]]:
        """Run a single evaluation step.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        batch : Any
            A batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        Mapping[str, Any] or None
            A dictionary of evaluation results for the batch or `None` if no
            batch results are available.

        """
        rank_zero_warn(
            f"`evaluation_step` must be implemented to use {self.__class__.__name__} for evaluation."
        )
        return None

    def on_evaluation_epoch_end(
        self, pl_module: pl.LightningModule
    ) -> Optional[Union[Mapping[str, Any]]]:
        """Run after the evaluation epoch.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.

        Returns
        -------
        Optional[Union[Mapping[str, Any]]]
            A dictionary of evaluation results or `None` if no results are available.
        """
