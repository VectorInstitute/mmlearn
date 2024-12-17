Getting Started
===============
*mmlearn* contains a collection of tools and utilities to help researchers and practitioners easily set up and run training
or evaluation experiments for multimodal representation learning methods. The toolkit is designed to be modular and extensible.
We aim to provide a high degree of flexibility in using existing methods, while also allowing users to easily add support
for new modalities of data, datasets, models and pretraining or evaluation methods.

Much of the power and flexibility of *mmlearn* comes from building on top of the `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_
framework and using `Hydra <https://hydra.cc/docs/intro/>`_ and `hydra-zen <https://mit-ll-responsible-ai.github.io/hydra-zen/>`_
for configuration management. Together, these tools make it easy to define and run experiments with different configurations,
and to scale up experiments to run on a SLURM cluster.

The goal of this guide is to give you a brief overview of what *mmlearn* is and how you can get started using it.

.. note::
   *mmlearn* currently only supports training and evaluation of encoder-only models.

   For more detailed information on the features and capabilities of *mmlearn*, please refer to the :doc:`API Reference <api>`.


Defining a Dataset
------------------
Datasets in *mmlearn* can be defined using PyTorch's :class:`~torch.utils.data.Dataset` or :class:`~torch.utils.data.IterableDataset`
classes. However, there are two additional requirements for datasets in *mmlearn*:

1. The dataset must return an instance of :class:`~mmlearn.datasets.core.example.Example` from the :meth:`~torch.utils.data.Dataset.__getitem__`
   method or the :meth:`~torch.utils.data.IterableDataset.__iter__` method.
2. The :class:`~mmlearn.datasets.core.example.Example` object returned by the dataset must contain the key ``'example_index'``
   and use modality-specific keys from the :class:`Modalities <mmlearn.datasets.core.modalities.ModalityRegistry>` registry
   to store the data.

**Example 1**: Defining a map-style dataset in *mmlearn*:

.. code-block:: python

   from torch.utils.data.dataset import Dataset

   from mmlearn.datasets.core import Example, Modalities
   from mmlearn.constants import EXAMPLE_INDEX_KEY


   class MyMapStyleDataset(Dataset[Example]):
      ...
      def __getitem__(self, idx: int) -> Example:
         ...
         return Example(
            {
               EXAMPLE_INDEX_KEY: idx,
               Modalities.TEXT.name: ...,
               Modalities.RGB.name: ...,
               Modalities.RGB.target: ...,
               Modalities.TEXT.mask: ...,
               ...
            }
         )

**Example 2**: Defining an iterable-style dataset in *mmlearn*:

.. code-block:: python

   from torch.utils.data.dataset import IterableDataset

   from mmlearn.datasets.core import Example, Modalities
   from mmlearn.constants import EXAMPLE_INDEX_KEY


   class MyIterableStyleDataset(IterableDataset[Example]):
      ...
      def __iter__(self) -> Generator[Example, None, None]:
         ...
         idx = 0
         for item in items:
            yield Example(
               {
                  EXAMPLE_INDEX_KEY: idx,
                  Modalities.TEXT.name: ...,
                  Modalities.AUDIO.name: ...,
                  Modalities.TEXT.mask: ...,
                  Modalities.AUDIO.mask: ...,
                  ...
               }
            )
            idx += 1

The :class:`~mmlearn.datasets.core.example.Example` class represents a single example in the dataset and all the attributes
associated with it. The class is an extension of the :class:`~collections.OrderedDict` class that provides attribute-style access
to the dictionary values and handles the creation of the ``'example_ids'`` tuple, combining the ``'example_index'`` and ``'dataset_index'``
values.

:py:data:`~mmlearn.datasets.core.modalities.Modalities` is an instance of :class:`~mmlearn.datasets.core.modalities.ModalityRegistry`
singleton class that serves as a global registry for all the modalities supported by *mmlearn*. It allows dot-style access
registered modalities and their properties. For example, the ``'RGB'`` modality can be accessed using  :py:data:`Modalities.RGB`
(returns string ``'rgb'``) and the ``'target'`` property of the ``'RGB'`` modality can be accessed using :py:data:`Modalities.RGB.target`
(returns the string ``'rgb_target'``). It also provides a method to register new modalities and their properties. For example,
the following code snippet shows how to register a new ``'DNA'`` modality:

.. code-block:: python

   from mmlearn.datasets.core import Modalities

   Modalities.register_modality("dna")


Creating a Model
----------------
Models in *mmlearn* are generally defined by extending PyTorch's :class:`nn.Module <torch.nn.Module>` class. The input to the model's
forward method should be a dictionary, where the keys are the names of the modalities and the values are the corresponding
(batched) tensors/data. The models must also return a list-like object where the first element is the last layer's output.

.. code-block:: python

   import torch
   from torch import nn

   from mmlearn.datasets.core import Modalities


   class MyTextEncoder(nn.Module):
      def __init__(self, input_dim: int, output_dim: int):
         super().__init__()
         self.encoder = ...

      def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
         out = self.encoder(
            inputs[Modalities.TEXT.name],
            inputs.get(
               "attention_mask", inputs.get(Modalities.TEXT.attention_mask, None)
            ),
         )
         return (out,)

Passing a dictionary of the (batched) inputs to the model's forward method makes it easier to reuse the same model for different
tasks.

Creating and Configuring a Project
----------------------------------
A project in *mmlearn* can be thought of as a collection of related experiments. Within a project, you can reuse components
from *mmlearn* (e.g., datasets, models, tasks) or define new ones and use them all together for experiments.

To create a new project, create a new directory following the structure below:

.. code-block:: bash

   my_project/
   ├── configs/
   │   ├── __init__.py
   │   └── experiment/
   │       ├── my_experiment.yaml
   ├── README.md (optional)
   ├── requirements.txt (optional)

The ``configs/`` directory contains all the configurations, both `structured configs <https://hydra.cc/docs/tutorials/structured_config/intro/>`_
and YAML config files for the experiments in the project. The ``configs/experiment/`` directory contains the `.yaml` files
for the experiments associated with the project. These `.yaml` files use the `Hydra configuration format <https://hydra.cc/docs/tutorials/basic/your_first_app/composition/>`_,
which also allows overriding the configuration options/values from the command line.

The ``__init__.py`` file in the ``configs/`` directory is required to make the ``configs/`` directory a Python package,
allowing hydra to compose configurations from `.yaml` files as well as structured configs from python modules. More on this
in the next section.

Optionally, you can also include a ``README.md`` file with a brief description of the project and a ``requirements.txt`` file
with the dependencies required to run the project.

Specifying Configurable Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the key features of the Hydra configuration system is the ability to compose configurations from multiple sources,
including the command line, `.yaml` files and structured configs from Python modules. `Structured Configs <https://hydra.cc/docs/tutorials/structured_config/intro/>`_
in Hydra use Python :func:`~dataclasses.dataclass` to define the configuration schema. This allows for both static and runtime type-checking
of the configuration. `Hydra-zen <https://mit-ll-responsible-ai.github.io/hydra-zen/>`_ extends Hydra to makes it easy
to dynamically generate dataclass-backed configurations for any class or function simply by adding a decorator to the class
or function.

*mmlearn* provides a pre-populated `config store <https://hydra.cc/docs/tutorials/structured_config/config_store/>`_,
:py:data:`~mmlearn.conf.external_store`, which can be used as a decorator to register configurable components. This config
store already contains configurations for common components like PyTorch :py:mod:`optimizers <torch.optim>`,
:py:mod:`learning rate schedulers <torch.optim.lr_scheduler>`, loss functions and samplers,
as well as PyTorch Lightning's Trainer :py:mod:`callbacks <lightning.pytorch.callbacks>` and :py:mod:`loggers <lightning.pytorch.loggers>`.
To dynamically add new configurable components to the store, simply add the :py:data:`~mmlearn.conf.external_store` decorator
to the class or function definition.

For example, the following code snippet shows how to register a new dataset class:

.. code-block:: python

   from torch.utils.data.dataset import Dataset

   from mmlearn.conf import external_store
   from mmlearn.constants import EXAMPLE_INDEX_KEY
   from mmlearn.datasets.core import Example, Modalities


   @external_store(group="datasets")
   class MyMapStyleDataset(Dataset[Example]):
      ...
      def __getitem__(self, idx: int) -> Example:
         ...
         return Example(
            {
               EXAMPLE_INDEX_KEY: idx,
               Modalities.TEXT.name: ...,
               Modalities.RGB.name: ...,
               Modalities.RGB.target: ...,
               Modalities.TEXT.mask: ...,
               ...
            }
         )

The :py:data:`~mmlearn.conf.external_store` decorator immediately add the class to the config store once the Python interpreter
loads the module containing the class. This is why the ``configs/`` directory must be a Python package and why modules
containing user-defined configurable components must be imported in the ``configs/__init__.py`` file.

The ``group`` argument specifies the `config group <https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/>`_
under which the configurable component will be registered. This allows users to easily reference the component in the
configurations using the group name and the class name. The available config groups in *mmlearn* are:

- ``datasets``: Contains all the dataset classes.
- ``datasets/masking``: Contains all the configurable classes and functions for masking input data.
- ``datasets/tokenizers``: Contains all the configurable classes and functions for converting raw inputs to tokens.
- ``datasets/transforms``: Contains all the configurable classes and functions for transforming input data.
- ``dataloader/sampler``: Contains all the dataloader sampler classes.
- ``modules/encoders``: Contains all the encoder modules.
- ``modules/layers``: For layers that can be used independent of the model.
- ``modules/losses``: Contains all the loss functions.
- ``modules/optimizers``: Contains all the optimizers.
- ``modules/lr_schedulers``: Contains all the learning rate schedulers.
- ``modules/metrics``: Contains all the evaluation metrics.
- ``tasks``: Contains all the task classes.
- ``trainer/callbacks``: Contains all the PyTorch Lightning Trainer callbacks.
- ``trainer/logger``: Contains all the PyTorch Lightning Trainer loggers.


The Base Configuration
~~~~~~~~~~~~~~~~~~~~~~~
The base configuration for all experiments in *mmlearn* are defined in the :class:`~mmlearn.conf.MMLearnConf`
dataclass. This serves as the base configuration for all experiments and can be extended to include additional configuration
options, following Hydra's `override syntax <https://hydra.cc/docs/advanced/override_grammar/basic/>`_.

The base configuration for *mmlearn* is shown below:

.. code-block:: yaml

   experiment_name: ???
   job_type: train
   seed: null
   datasets:
      train: null
      val: null
      test: null
   dataloader:
      train:
         _target_: torch.utils.data.dataloader.DataLoader
         _convert_: object
         dataset: ???
         batch_size: 1
         shuffle: null
         sampler: null
         batch_sampler: null
         num_workers: 0
         collate_fn:
            _target_: mmlearn.datasets.core.data_collator.DefaultDataCollator
            batch_processors: null
         pin_memory: true
         drop_last: false
         timeout: 0.0
         worker_init_fn: null
         multiprocessing_context: null
         generator: null
         prefetch_factor: null
         persistent_workers: false
         pin_memory_device: ''
      val:
         _target_: torch.utils.data.dataloader.DataLoader
         _convert_: object
         dataset: ???
         batch_size: 1
         shuffle: null
         sampler: null
         batch_sampler: null
         num_workers: 0
         collate_fn:
            _target_: mmlearn.datasets.core.data_collator.DefaultDataCollator
            batch_processors: null
         pin_memory: true
         drop_last: false
         timeout: 0.0
         worker_init_fn: null
         multiprocessing_context: null
         generator: null
         prefetch_factor: null
         persistent_workers: false
         pin_memory_device: ''
      test:
         _target_: torch.utils.data.dataloader.DataLoader
         _convert_: object
         dataset: ???
         batch_size: 1
         shuffle: null
         sampler: null
         batch_sampler: null
         num_workers: 0
         collate_fn:
            _target_: mmlearn.datasets.core.data_collator.DefaultDataCollator
            batch_processors: null
         pin_memory: true
         drop_last: false
         timeout: 0.0
         worker_init_fn: null
         multiprocessing_context: null
         generator: null
         prefetch_factor: null
         persistent_workers: false
         pin_memory_device: ''
   task: ???
   trainer:
      _target_: lightning.pytorch.trainer.trainer.Trainer
      accelerator: auto
      strategy: auto
      devices: auto
      num_nodes: 1
      precision: null
      logger: null
      callbacks: null
      fast_dev_run: false
      max_epochs: null
      min_epochs: null
      max_steps: -1
      min_steps: null
      max_time: null
      limit_train_batches: null
      limit_val_batches: null
      limit_test_batches: null
      limit_predict_batches: null
      overfit_batches: 0.0
      val_check_interval: null
      check_val_every_n_epoch: 1
      num_sanity_val_steps: null
      log_every_n_steps: null
      enable_checkpointing: true
      enable_progress_bar: true
      enable_model_summary: true
      accumulate_grad_batches: 1
      gradient_clip_val: null
      gradient_clip_algorithm: null
      deterministic: null
      benchmark: null
      inference_mode: true
      use_distributed_sampler: true
      profiler: null
      detect_anomaly: false
      barebones: false
      plugins: null
      sync_batchnorm: false
      reload_dataloaders_every_n_epochs: 0
      default_root_dir: ${hydra:runtime.output_dir}/checkpoints
   tags:
      - ${experiment_name}
   resume_from_checkpoint: null
   strict_loading: true
   torch_compile_kwargs:
      disable: true
      fullgraph: false
      dynamic: null
      backend: inductor
      mode: null
      options: null

The config keys with a value of ``???`` are placeholders that must be overridden in the experiment configurations. While
the ``dataset`` key in the ``dataloader`` group is also a placeholder, it should not be provided as it will be automatically
filled in from the ``datasets`` group.


Running an Experiment
---------------------
To run an experiment locally, use the following command:

.. code:: bash

   mmlearn_run 'hydra.searchpath=[pkg://path.to.my_project.configs]' \
      +experiment=my_experiment \
      experiment_name=my_experiment_name

.. tip::
   You can see the full config for an experiment without running it by adding the ``--help`` flag to the command.

   .. code:: bash

      mmlearn_run 'hydra.searchpath=[pkg://path.to.my_project.configs]' \
         +experiment=my_experiment \
         experiment_name=my_experiment_name \
         task=my_task \ # required for the command to run
         --help

To run the experiment on a SLURM cluster, use the following command:

.. code:: bash

   mmlearn_run --multirun \
      hydra.launcher.mem_per_cpu=5G \
      hydra.launcher.qos=your_qos \
      hydra.launcher.partition=your_partition \
      hydra.launcher.gres=gpu:4 \
      hydra.launcher.cpus_per_task=8 \
      hydra.launcher.tasks_per_node=4 \
      hydra.launcher.nodes=1 \
      hydra.launcher.stderr_to_stdout=true \
      hydra.launcher.timeout_min=720 \
      'hydra.searchpath=[pkg://path.to.my_project.configs]' \
      +experiment=my_experiment \
      experiment_name=my_experiment_name

This uses the `submitit launcher <https://hydra.cc/docs/plugins/submitit_launcher/>`_ plugin built into Hydra to submit
the experiment to the SLURM scheduler with the specified resources.

.. note::
   After the job is submitted, it is okay to cancel the program with ``Ctrl+C``. The job will continue running on
   the cluster. You can also add ``&`` at the end of the command to run it in the background.
