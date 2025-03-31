"""Exponential Moving Average (EMA) module."""

import copy
from typing import Any, Optional, Set, Union

import torch
from lightning.fabric.utilities import rank_zero_warn


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) for the input model.

    At each step the parameter of the EMA model is updates as the weighted average
    of the model's parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to apply EMA to.
    ema_decay : float
        The initial decay value for EMA.
    ema_end_decay : float
        The final decay value for EMA.
    ema_anneal_end_step : int
        The number of steps to anneal the decay from ``ema_decay`` to ``ema_end_decay``.
    skip_keys : Optional[Union[list[str], Set[str]]], optional, default=None
        The keys to skip in the EMA update. These parameters will be copied directly
        from the model to the EMA model.

    Raises
    ------
    RuntimeError
        If a deep copy of the model cannot be created.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ema_decay: float,
        ema_end_decay: float,
        ema_anneal_end_step: int,
        skip_keys: Optional[Union[list[str], Set[str]]] = None,
    ) -> None:
        self.model = self.deepcopy_model(model)

        self.skip_keys: Union[list[str], set[str]] = skip_keys or set()
        self.num_updates = 0
        self.decay = ema_decay  # stores the current decay value
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step

        self._model_configured = False

    @staticmethod
    def deepcopy_model(model: torch.nn.Module) -> torch.nn.Module:
        """Deep copy the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to copy.

        Returns
        -------
        torch.nn.Module
            The copied model.

        Raises
        ------
        RuntimeError
            If the model cannot be copied.
        """
        try:
            return copy.deepcopy(model)
        except RuntimeError as e:
            raise RuntimeError("Unable to copy the model ", e) from e

    @staticmethod
    def get_annealed_rate(
        start: float,
        end: float,
        curr_step: int,
        total_steps: int,
    ) -> float:
        """Calculate EMA annealing rate."""
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining

    def configure_model(self, device_id: Union[int, torch.device]) -> None:
        """Configure the model for EMA."""
        if self._model_configured:
            return

        self.model.requires_grad_(False)
        self.model.to(device_id)

        self._model_configured = True

    def step(self, new_model: torch.nn.Module) -> None:
        """Perform single EMA update step."""
        if not self._model_configured:
            raise RuntimeError(
                "Model is not configured for EMA. Call `configure_model` first."
            )

        self._update_weights(new_model)
        self._update_ema_decay()

    def restore(self, model: torch.nn.Module) -> torch.nn.Module:
        """Reassign weights from another model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load weights from.

        Returns
        -------
        torch.nn.Module
            model with new weights
        """
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model

    def state_dict(self) -> dict[str, Any]:
        """Return the state dict of the model."""
        return self.model.state_dict()  # type: ignore[no-any-return]

    @torch.no_grad()  # type: ignore[misc]
    def _update_weights(self, new_model: torch.nn.Module) -> None:
        if self.decay < 1:
            ema_state_dict = {}
            ema_params = self.model.state_dict()

            for key, param in new_model.state_dict().items():
                ema_param = ema_params[key].float()

                if param.shape != ema_param.shape:
                    raise ValueError(
                        "Incompatible tensor shapes between student param and teacher param"
                        + "{} vs. {}".format(param.shape, ema_param.shape)
                    )

                if key in self.skip_keys or not param.requires_grad:
                    ema_param = param.to(dtype=ema_param.dtype).clone()
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.to(dtype=ema_param.dtype),
                        alpha=1 - self.decay,
                    )
                ema_state_dict[key] = ema_param

            self.model.load_state_dict(ema_state_dict, strict=False)
            self.num_updates += 1
        else:
            rank_zero_warn(
                "Exponential Moving Average decay is 1.0, no update is applied to the model.",
                stacklevel=1,
                category=UserWarning,
            )

    def _update_ema_decay(self) -> None:
        if self.ema_decay != self.ema_end_decay:
            if self.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.num_updates,
                    self.ema_anneal_end_step,
                )
            self.decay = decay
