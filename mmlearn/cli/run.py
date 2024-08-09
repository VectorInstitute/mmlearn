"""Main entry point for training and evaluation."""

import copy
import logging
from typing import Optional

import hydra
import lightning as L  # noqa: N812
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from mmlearn.cli._instantiators import (
    instantiate_callbacks,
    instantiate_datasets,
    instantiate_loggers,
    instantiate_sampler,
)
from mmlearn.conf import JobType, MMLearnConf, hydra_main
from mmlearn.datasets.core import *  # noqa: F403
from mmlearn.datasets.processors import *  # noqa: F403
from mmlearn.modules.encoders import *  # noqa: F403
from mmlearn.modules.layers import *  # noqa: F403
from mmlearn.modules.losses import *  # noqa: F403
from mmlearn.modules.lr_schedulers import *  # noqa: F403
from mmlearn.modules.metrics import *  # noqa: F403
from mmlearn.tasks import *  # noqa: F403


logger = logging.getLogger(__package__)


@hydra_main(
    config_path="pkg://mmlearn.conf", config_name="base_config", version_base=None
)
def main(cfg: MMLearnConf) -> None:  # noqa: PLR0912
    """Entry point for training or evaluation."""
    cfg_copy = copy.deepcopy(cfg)  # copy of the config for logging

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # setup trainer first so that we can get some variables for distributed training
    callbacks = instantiate_callbacks(cfg.trainer.get("callbacks"))
    cfg.trainer["callbacks"] = None  # will be replaced with the instantiated object
    loggers = instantiate_loggers(cfg.trainer.get("logger"))
    cfg.trainer["logger"] = None
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers, _convert_="all"
    )
    assert isinstance(
        trainer, Trainer
    ), "Trainer must be an instance of `lightning.pytorch.trainer.Trainer`"

    if rank_zero_only.rank == 0 and loggers is not None:  # update wandb config
        for trainer_logger in loggers:
            if isinstance(trainer_logger, WandbLogger):
                trainer_logger.experiment.config.update(
                    OmegaConf.to_container(cfg_copy, resolve=True, enum_to_str=True),
                    allow_val_change=True,
                )
    trainer.print(OmegaConf.to_yaml(cfg_copy, resolve=True))

    requires_distributed_sampler = (
        trainer.distributed_sampler_kwargs is not None
        and trainer._accelerator_connector.use_distributed_sampler
    )
    if requires_distributed_sampler:  # we handle distributed samplers
        trainer._accelerator_connector.use_distributed_sampler = False

    # prepare dataloaders
    if cfg.job_type == JobType.train:
        train_dataset = instantiate_datasets(cfg.datasets.train)
        assert (
            train_dataset is not None
        ), "Train dataset (`cfg.datasets.train`) is required for training."

        train_sampler = instantiate_sampler(
            cfg.dataloader.train.get("sampler"),
            train_dataset,
            requires_distributed_sampler=requires_distributed_sampler,
            distributed_sampler_kwargs=trainer.distributed_sampler_kwargs,
        )
        cfg.dataloader.train["sampler"] = None  # replaced with the instantiated object
        train_loader: DataLoader = hydra.utils.instantiate(
            cfg.dataloader.train, dataset=train_dataset, sampler=train_sampler
        )

        val_loader: Optional[DataLoader] = None
        val_dataset = instantiate_datasets(cfg.datasets.val)
        if val_dataset is not None:
            val_sampler = instantiate_sampler(
                cfg.dataloader.val.get("sampler"),
                val_dataset,
                requires_distributed_sampler=requires_distributed_sampler,
                distributed_sampler_kwargs=trainer.distributed_sampler_kwargs,
            )
            cfg.dataloader.val["sampler"] = None
            val_loader = hydra.utils.instantiate(
                cfg.dataloader.val, dataset=val_dataset, sampler=val_sampler
            )
    else:
        test_dataset = instantiate_datasets(cfg.datasets.test)
        assert (
            test_dataset is not None
        ), "Test dataset (`cfg.datasets.test`) is required for evaluation."

        test_sampler = instantiate_sampler(
            cfg.dataloader.test.get("sampler"),
            test_dataset,
            requires_distributed_sampler=requires_distributed_sampler,
            distributed_sampler_kwargs=trainer.distributed_sampler_kwargs,
        )
        cfg.dataloader.test["sampler"] = None
        test_loader = hydra.utils.instantiate(
            cfg.dataloader.test, dataset=test_dataset, sampler=test_sampler
        )

    # setup task module
    if cfg.task is None or "_target_" not in cfg.task:
        raise ValueError(
            "Expected a non-empty config for `cfg.task` with a `_target_` key. "
            f"But got: {cfg.task}"
        )
    logger.info(f"Instantiating task module: {cfg.task['_target_']}")
    model: L.LightningModule = hydra.utils.instantiate(cfg.task, _convert_="partial")
    assert isinstance(model, L.LightningModule), "Task must be a `LightningModule`"
    model.strict_loading = cfg.strict_loading

    # compile model
    model = torch.compile(model, **OmegaConf.to_object(cfg.torch_compile_kwargs))

    if cfg.job_type == JobType.train:
        trainer.fit(
            model, train_loader, val_loader, ckpt_path=cfg.resume_from_checkpoint
        )
    elif cfg.job_type == JobType.eval:
        trainer.test(model, test_loader, ckpt_path=cfg.resume_from_checkpoint)


if __name__ == "__main__":
    main()
