# mmlearn
[![code checks](https://github.com/VectorInstitute/mmlearn/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/mmlearn/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/mmlearn/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/mmlearn/actions/workflows/integration_tests.yml)
[![license](https://img.shields.io/github/license/VectorInstitute/mmlearn.svg)](https://github.com/VectorInstitute/mmlearn/blob/main/LICENSE)

This project aims at enabling the evaluation of existing multimodal representation learning methods, as well as facilitating
experimentation and research for new techniques.

## Quick Start
### Installation
#### Prerequisites
The library requires Python 3.9 or later. We recommend using a virtual environment to manage dependencies. You can create
a virtual environment using the following command:
```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

#### Installing binaries
To install the pre-built binaries, run:
```bash
python3 -m pip install mmlearn
```

#### Building from source
To install the library from source, run:

```bash
git clone https://github.com/VectorInstitute/mmlearn.git
cd mmlearn
python3 -m pip install -e .
```

### Running Experiments
To run an experiment, create a folder with a similar structure as the [`configs`](configs/) folder.
Then, use the `mmlearn_run` command to run the experiment as defined in a `.yaml` file under the `experiment` folder, like so:
```bash
mmlearn_run --config-dir /path/to/config/dir +experiment=<name_of_experiment_config> experiment=your_experiment_name
```
Notice that the config directory refers to the top-level directory containing the `experiment` folder. The experiment
name is the name of the `.yaml` file under the `experiment` folder, without the extension.

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations, so you can override any configuration parameter
from the command line. To see the available options and other information, run:
```bash
mmlearn_run --config-dir /path/to/config/dir +experiment=<name_of_experiment> --help
```

By default, the `mmlearn_run` command will run the experiment locally. To run the experiment on a SLURM cluster, we use
the [submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher/) plugin built into Hydra. The following is an example
of how to run an experiment on a SLURM cluster:
```bash
mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.qos=your_qos hydra.launcher.partition=your_partition hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' --config-dir /path/to/config/dir +experiment=<name_of_experiment_config> experiment=your_experiment_name
```
This will submit a job to the SLURM cluster with the specified resources.

**Note**: After the job is submitted, it is okay to cancel the program with `Ctrl+C`. The job will continue running on
the cluster. You can also add `&` at the end of the command to run it in the background.


## Summary of Implemented Methods
<table>
<tr>
<th style="text-align: left; width: 250px"> Pretraining Methods </th>
<th style="text-align: center"> Notes </th>
</tr>
<tr>
<td>

Contrastive Pretraining
</td>
<td>
Uses the contrastive loss to align the representations from <i>N</i> modalities. Supports sharing of encoders, projection heads
or postprocessing modules (e.g. logit/temperature scaling) across modalities. Also supports multi-task learning with auxiliary
unimodal tasks applied to specific modalities.
</td>
</tr>
<tr>
<th style="text-align: left; width: 250px"> Evaluation Methods </th>
<th style="text-align: center"> Notes </th>
</tr>
<tr>
<td>

Zero-shot Cross-modal Retrieval
</td>
<td>
Evaluates the quality of the learned representations in retrieving the <i>k</i> most similar examples from a different modality,
using recall@k metric. This is applicable to any number of pairs of modalities at once, depending on memory constraints.
</td>
</tr>
</table>

## Components
### Datasets
Every dataset object must return an instance of [`Example`](mmlearn/datasets/core/example.py) with one or more keys/attributes
corresponding to a modality name as specified in the [`Modalities registry`](mmlearn/datasets/core/modalities.py).
The `Example` object must also include an `example_index` attribute/key, which is used, in addition to the dataset index,
to uniquely identify the example.

<details>
<summary><b>CombinedDataset</b></summary>

The [`CombinedDataset`](mmlearn/datasets/core/combined_dataset.py) object is used to combine multiple datasets into one. It
accepts an iterable of `torch.utils.data.Dataset` and/or `torch.utils.data.IterableDataset` objects and returns an `Example`
object from one of the datasets, given an index. Conceptually, the `CombinedDataset` object is a concatenation of the
datasets in the input iterable, so the given index can be mapped to a specific dataset based on the size of the datasets.
As iterable-style datasets do not support random access, the examples from these datasets are returned in order as they
are iterated over.

The `CombinedDataset` object also adds a `dataset_index` attribute to the `Example` object, corresponding to the index of
the dataset in the input iterable. Every example returned by the `CombinedDataset` will have an `example_ids` attribute,
which is instance of `Example` containing the same keys/attributes as the original example, with the exception of the
`example_index` and `dataset_index` attributes, with values being a tensor of the `dataset_index` and `example_index`.
</details>

### Dataloading
When dealing with multiple datasets with different modalities, the default `collate_fn` of `torch.utils.data.DataLoader`
may not work, as it assumes that all examples have the same keys/attributes. In that case, the [`collate_example_list`](mmlearn/datasets/core/example.py)
function can be used as the `collate_fn` argument of `torch.utils.data.DataLoader`. This function takes a list of `Example`
objects and returns a dictionary of tensors, with all the keys/attributes of the `Example` objects.

## Contributing

If you are interested in contributing to the library, please see [CONTRIBUTING.MD](CONTRIBUTING.MD). This file contains
many details around contributing to the code base, including are development practices, code checks, tests, and more.
