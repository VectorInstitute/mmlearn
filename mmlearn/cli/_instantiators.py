"""Methods for instantiating objects from config.

This module is for objects that require more complex instantiation logic than
what is provided by Hydra's `instantiate` method.
"""

import logging
from typing import Any, Optional, Union

import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    IterableDataset,
)
from torch.utils.data.sampler import Sampler

from mmlearn.datasets.core.combined_dataset import CombinedDataset


logger = logging.getLogger(__package__)


def instantiate_datasets(
    cfg: Optional[DictConfig],
) -> Optional[CombinedDataset]:
    """Instantiate datasets from config.

    Parameters
    ----------
    cfg : DictConfig, optional
        A DictConfig object containing dataset configurations.

    Returns
    -------
    Optional[CombinedDataset]
        The instantiated dataset(s), wrapped in a :py:class:`CombinedDataset` object.
        If no dataset configurations are provided, returns ``None``.
    """
    if cfg is None:
        return None

    datasets: list[Union[Dataset, IterableDataset]] = []
    if "_target_" in cfg:  # single dataset
        logger.info(f"Instantiating dataset: {cfg._target_}")
        datasets.append(hydra.utils.instantiate(cfg, _convert_="partial"))
    else:  # multiple datasets
        for _, ds_conf in cfg.items():
            if "_target_" in ds_conf:
                logger.info(f"Instantiating dataset: {ds_conf._target_}")
                datasets.append(hydra.utils.instantiate(ds_conf, _convert_="partial"))
            else:
                logger.warn(
                    f"Skipping dataset configuration: {ds_conf}. No `_target_` key found."
                )

    return CombinedDataset(datasets) if datasets else None


def instantiate_sampler(
    cfg: Optional[DictConfig],
    dataset: Optional[Union[CombinedDataset, Dataset, IterableDataset]],
    requires_distributed_sampler: bool,
    distributed_sampler_kwargs: Optional[dict[str, Any]],
) -> Optional[Sampler]:
    """Instantiate sampler from config.

    Parameters
    ----------
    cfg : DictConfig, optional
        The configuration for the sampler.
    dataset : Union[CombinedDataset, Dataset, IterableDataset], optional
        The dataset for which to instantiate the sampler.
    requires_distributed_sampler : bool
        Whether a distributed sampler is required. This is typically True when
        the lightning trainer is using a Parallel strategy and the
        `use_distributed_sampler` flag is set to True.
    distributed_sampler_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to the distributed sampler.

    Returns
    -------
    Optional[Sampler]
        The instantiated sampler or ``None`` if no sampler has been instantiated.
    """
    if distributed_sampler_kwargs is None:
        distributed_sampler_kwargs = {}

    sampler: Optional[Sampler] = None
    if cfg is not None:
        kwargs = (
            {"data_source": dataset}
            if "data_source" in cfg.keys()  # noqa: SIM118
            else {"dataset": dataset}
        )
        if (
            requires_distributed_sampler
            or "CombinedDatasetRatioSampler" in cfg._target_
        ):
            kwargs.update(distributed_sampler_kwargs)

        sampler = hydra.utils.instantiate(cfg, **kwargs)
        assert isinstance(sampler, Sampler), (
            f"Expected a `torch.utils.data.Sampler` object but got {type(sampler)}."
        )

    if sampler is None and requires_distributed_sampler:
        sampler = DistributedSampler(dataset, **distributed_sampler_kwargs)

    return sampler


def instantiate_callbacks(cfg: Optional[DictConfig]) -> Optional[list[Callback]]:
    """Instantiate callbacks from config.

    Parameters
    ----------
    cfg : DictConfig, optional
        A DictConfig object containing callback configurations.

    Returns
    -------
    Optional[list[Callback]]
        A list of instantiated callbacks or ``None`` if no callback configurations
        are provided.

    Raises
    ------
    TypeError
        If the instantiated object is not a pytorch lightning ``Callback``
    """
    if cfg is None:
        return None

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Expected `cfg` to be an instance of `DictConfig` but got {type(cfg)}."
        )

    callbacks: list[Callback] = []
    for _, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            obj = hydra.utils.instantiate(cb_conf, _convert_="partial")
            if not isinstance(obj, Callback):
                raise TypeError(
                    f"Expected a pytorch lightning `Callback` object but got {type(obj)}."
                )
            callbacks.append(obj)

    return callbacks


def instantiate_loggers(cfg: Optional[DictConfig]) -> Optional[list[Logger]]:
    """Instantiate loggers from config.

    Parameters
    ----------
    cfg : DictConfig, optional
        A DictConfig object containing logger configurations.

    Returns
    -------
    Optional[list[Logger]]
        A list of instantiated loggers or ``None`` if no configurations are provided.

    Raises
    ------
    TypeError
        If the instantiated object is not a pytorch lightning ``Logger`` or the
        configuration is not a ``DictConfig``.
    """
    if cfg is None:
        return None

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Expected `cfg` to be an instance of `DictConfig` but got {type(cfg)}."
        )

    logger: list[Logger] = []
    for _, lg_conf in cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            obj = hydra.utils.instantiate(lg_conf, _convert_="partial")
            if not isinstance(obj, Logger):
                raise TypeError(
                    f"Expected a pytorch lightning `Logger` object but got {type(obj)}."
                )
            logger.append(obj)

    return logger
