"""Hydra/Hydra-zen-based configurations."""

import functools
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

import hydra
import lightning.pytorch.callbacks as lightning_callbacks
import lightning.pytorch.loggers as lightning_loggers
import lightning.pytorch.trainer as lightning_trainer
import torch.nn.modules.loss as torch_losses
import torch.optim as torch_optim
import torch.utils.data
import torch.utils.data.distributed
from hydra.conf import HelpConf, HydraConf, JobConf, RunDir, SweepDir
from hydra.core.config_store import ConfigStore
from hydra.main import _UNSPECIFIED_
from hydra.types import TaskFunction
from hydra_zen import ZenStore, builds, store
from lightning.pytorch.loggers.wandb import _WANDB_AVAILABLE
from omegaconf import II, MISSING, SI, DictConfig

from mmlearn.datasets.core.data_collator import DefaultDataCollator


def _get_default_ckpt_dir() -> Any:
    """Get the default checkpoint directory."""
    return SI("${hydra:runtime.output_dir}/checkpoints")


_DataLoaderConf = builds(
    torch.utils.data.DataLoader,
    populate_full_signature=True,
    dataset=MISSING,
    pin_memory=True,
    collate_fn=DefaultDataCollator(),
    hydra_convert="object",
)


class JobType(str, Enum):
    """Type of the job."""

    train = "train"
    eval = "eval"


@dataclass
class DatasetConf:
    """Configuration template for the datasets."""

    #: Configuration for the training dataset.
    train: Optional[Any] = field(
        default=None,
        metadata={"help": "Configuration for the training dataset."},
    )
    #: Configuration for the validation dataset.
    val: Optional[Any] = field(
        default=None, metadata={"help": "Configuration for the validation dataset."}
    )
    #: Configuration for the test dataset.
    test: Optional[Any] = field(
        default=None,
        metadata={"help": "Configuration for the test dataset."},
    )


@dataclass
class DataLoaderConf:
    """Configuration for the dataloader."""

    #: Configuration for the training dataloader.
    train: Any = field(
        default_factory=_DataLoaderConf,
        metadata={"help": "Configuration for the training dataloader."},
    )
    #: Configuration for the validation dataloader.
    val: Any = field(
        default_factory=_DataLoaderConf,
        metadata={"help": "Configuration for the validation dataloader."},
    )
    #: Configuration for the test dataloader.
    test: Any = field(
        default_factory=_DataLoaderConf,
        metadata={"help": "Configuration for the test dataloader."},
    )


@dataclass
class MMLearnConf:
    """Top-level configuration for mmlearn experiments."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",  # See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
            {"task": MISSING},
            {"override hydra/launcher": "submitit_slurm"},
        ]
    )
    #: Name of the experiment. This must be specified for any experiment to run.
    experiment_name: str = field(default=MISSING)
    #: Type of the job.
    job_type: JobType = field(default=JobType.train)
    #: Seed for the random number generators. This is set for Python, Numpy and PyTorch,
    #: including the workers in PyTorch Dataloaders.
    seed: Optional[int] = field(default=None)
    #: Configuration for the datasets.
    datasets: DatasetConf = field(default_factory=DatasetConf)
    #: Configuration for the dataloaders.
    dataloader: DataLoaderConf = field(default_factory=DataLoaderConf)
    #: Configuration for the task. This is required to run any experiment.
    task: Any = field(default=MISSING)
    #: Configuration for the trainer. The options here are the same as in
    #: :py:class:`~lightning.pytorch.trainer.trainer.Trainer`
    trainer: Any = field(
        default_factory=builds(
            lightning_trainer.Trainer,
            populate_full_signature=True,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True,
            default_root_dir=_get_default_ckpt_dir(),
        )
    )
    #: Tags for the experiment. This is useful for `wandb <https://docs.wandb.ai/ref/python/init>`_
    #: logging.
    tags: Optional[list[str]] = field(default_factory=lambda: [II("experiment_name")])
    #: Path to the checkpoint to resume training from.
    resume_from_checkpoint: Optional[Path] = field(default=None)
    #: Whether to strictly enforce loading of model weights i.e. `strict=True` in
    #: :py:meth:`~lightning.pytorch.core.module.LightningModule.load_from_checkpoint`.
    strict_loading: bool = field(default=True)
    #: Configuration for torch.compile. These are essentially the same as the
    #: arguments for :py:func:`torch.compile`.
    torch_compile_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "disable": True,
            "fullgraph": False,
            "dynamic": None,
            "backend": "inductor",
            "mode": None,
            "options": None,
        }
    )
    #: Hydra configuration.
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            searchpath=["pkg://mmlearn.conf"],
            run=RunDir(
                dir=SI("./outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}")
            ),
            sweep=SweepDir(
                dir=SI("./outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"),
                subdir=SI("${hydra.job.num}_${hydra.job.id}"),
            ),
            help=HelpConf(
                app_name="mmlearn",
                header="mmlearn: A modular framework for research on multimodal representation learning.",
            ),
            job=JobConf(
                name=II("experiment_name"),
                env_set={
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                    "HYDRA_FULL_ERROR": "1",
                },
            ),
        )
    )


cs = ConfigStore.instance()

cs.store(
    name="base_config",
    node=MMLearnConf,
    package="_global_",
    provider="mmlearn",
)


#################### External Modules ####################
#: A custom ZenStore object that will immediately add entries to Hydra's global
#: config store as soon as they are registered. Use this as a decorator for
#: newly-defined configurable functions/classes outside the main mmlearn package.
external_store = ZenStore(name="external_store", deferred_hydra_store=False)


def register_external_modules(
    module: ModuleType,
    group: str,
    name: Optional[str] = None,
    package: Optional[str] = None,
    provider: Optional[str] = None,
    base_cls: Optional[type] = None,
    ignore_cls: Optional[list[type]] = None,
    ignore_prefix: Optional[str] = None,
    **kwargs_for_builds: Any,
) -> None:
    """Add all classes in an external module to a ZenStore.

    Parameters
    ----------
    module : ModuleType
        The module to add classes from.
    group : str
        The config group to add the classes to.
    name : Optional[str], optional, default=None
        The name to give to the dynamically-generated configs. If `None`, the
        class name is used.
    package : Optional[str], optional, default=None
        The package to add the configs to.
    provider : Optional[str], optional, default=None
        The provider to add the configs to.
    base_cls : Optional[type], optional, default=None
        The base class to filter classes by. The base class is also excluded from
        the configs.
    ignore_cls : Optional[list[type]], optional, default=None
        list of classes to ignore.
    ignore_prefix : Optional[str], optional, default=None
        Ignore classes whose names start with this prefix.
    kwargs_for_builds : Any
        Additional keyword arguments to pass to ``hydra_zen.builds``.

    """
    for key, cls in module.__dict__.items():
        if (
            isinstance(cls, type)
            and (base_cls is None or issubclass(cls, base_cls))
            and cls != base_cls
            and (ignore_cls is None or cls not in ignore_cls)
            and (ignore_prefix is None or not key.startswith(ignore_prefix))
        ):
            external_store(
                builds(cls, populate_full_signature=True, **kwargs_for_builds),
                name=name or key,
                group=group,
                package=package,
                provider=provider,
            )


register_external_modules(
    torch_optim,
    group="modules/optimizers",
    provider="torch",
    base_cls=torch_optim.Optimizer,
    zen_partial=True,
)

# NOTE: learning rate schedulers require partial instantiation (for adding the optimizer
# at runtime) and most of them have more than one required argument. When using
# `zen_partial=True`, hydra-zen will remove all arguments that don't have a default
# value. This is why we need to manually specify the required arguments for each
# learning rate scheduler.
external_store(
    builds(
        torch_optim.lr_scheduler.StepLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        step_size=MISSING,
    ),
    name="StepLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.MultiStepLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        milestones=MISSING,
    ),
    name="MultiStepLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.ExponentialLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        gamma=MISSING,
    ),
    name="ExponentialLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.CosineAnnealingLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        T_max=MISSING,
    ),
    name="CosineAnnealingLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.CyclicLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        base_lr=MISSING,
        max_lr=MISSING,
    ),
    name="CyclicLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.OneCycleLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        max_lr=MISSING,
    ),
    name="OneCycleLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.ReduceLROnPlateau,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
    ),
    name="ReduceLROnPlateau",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.LinearLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
    ),
    name="LinearLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.PolynomialLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
    ),
    name="PolynomialLR",
    group="modules/lr_schedulers",
    provider="torch",
)
external_store(
    builds(
        torch_optim.lr_scheduler.CosineAnnealingWarmRestarts,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        T_0=MISSING,
    ),
    name="CosineAnnealingWarmRestarts",
    group="modules/lr_schedulers",
    provider="torch",
)

register_external_modules(
    torch_losses,
    group="modules/losses",
    provider="torch",
    base_cls=torch_losses._Loss,
    ignore_prefix="_",
)

# manually include samplers whose first argument is a dataset object
external_store(
    torch.utils.data.RandomSampler,
    group="dataloader/sampler",
    provider="torch",
)
external_store(
    torch.utils.data.SequentialSampler,
    group="dataloader/sampler",
    provider="torch",
)
external_store(
    torch.utils.data.distributed.DistributedSampler,
    group="dataloader/sampler",
    provider="torch",
)

# NOTE: as of v2.3.3, the `device` filled for StochasticWeightAveraging has a default
# value of torch.device("cpu"), which is not a serializable type.
# ModelCheckpoint is configured separately so as to set some reasonable defaults.
register_external_modules(
    lightning_callbacks,
    group="trainer/callbacks",
    provider="lightning",
    base_cls=lightning_callbacks.Callback,
    ignore_cls=[
        lightning_callbacks.StochasticWeightAveraging,
        lightning_callbacks.ModelCheckpoint,
    ],
)
external_store(
    builds(
        lightning_callbacks.ModelCheckpoint,
        populate_full_signature=True,
        dirpath=_get_default_ckpt_dir(),
    ),
    name="ModelCheckpoint",
    group="trainer/callbacks",
    provider="lightning",
)

if _WANDB_AVAILABLE:
    register_external_modules(
        lightning_loggers,
        group="trainer/logger",
        provider="lightning",
        base_cls=lightning_loggers.Logger,
        ignore_cls=[lightning_loggers.WandbLogger],
    )
    external_store(
        builds(
            lightning_loggers.WandbLogger,
            populate_full_signature=True,
            name=II("experiment_name"),
            save_dir=SI("${hydra:runtime.output_dir}"),
            dir=SI("${hydra:runtime.output_dir}"),
            project=SI("${oc.env:WANDB_PROJECT}"),
            resume="allow",
            tags=II("tags"),
            job_type=II("job_type"),
        ),
        name="WandbLogger",
        group="trainer/logger",
        provider="lightning",
    )
else:
    warnings.warn(
        "wandb is not available. Skipping registration of 'trainer/logger/WandbLogger'.",
        stacklevel=1,
    )


#################### Custom Hydra Main Decorator ####################
def _hydra_main(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """Add hydra_zen configs to hydra's global config store.

    Custom hydra main decorator that adds hydra_zen configs to global store.

    Parameters
    ----------
    config_path :
        The config path, a directory where Hydra will search for config files.
        This path is added to Hydra's searchpath. Relative paths are interpreted
        relative to the declaring python file. Alternatively, you can use the prefix
        `pkg://` to specify a python package to add to the searchpath.
        If config_path is None no directory is added to the Config search path.
    config_name :
        The name of the config (usually the file name without the .yaml extension)
    """

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            store.add_to_hydra_store()
            return hydra.main(
                config_path=config_path,
                config_name=config_name,
                version_base=version_base,
            )(task_function)(cfg_passthrough)

        return decorated_main

    return main_decorator
