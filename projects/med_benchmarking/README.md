## Benchmarking CLIP-style Methods on Medical Data
Prior to running any experiments under this project, please install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
**NOTE**: It is assumed that the requirements for the `mmlearn` package have already been installed in a virtual environment.
If not, please refer to the README file in the `mmlearn` package for installation instructions.

Also, please make sure to set the following environment variables:
```bash
export MIMICIVCXR_ROOT_DIR=/path/to/mimic-cxr/data
export PMCOA_ROOT_DIR=/path/to/pmc_oa/data
export QUILT_ROOT_DIR=/path/to/quilt/data
export ROCO_ROOT_DIR=/path/to/roco/data
```

If you are running an experiment with the MedVQA dataset, please also set the following environment variables:
```bash
export PATHVQA_ROOT_DIR=/path/to/pathvqa/data
export VQARAD_ROOT_DIR=/path/to/vqarad/data
```

To run an experiment (pretraining), use the following command:

**To Run Locally**:
```bash
mmlearn_run 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=test
```

**To Run on a SLURM Cluster**:
```bash
mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.qos=your_qos hydra.launcher.partition=your_partition hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=test
```

To run zero-shot retrieval evaluation on a pretrained model locally (on the ROCO dataset, as an example), use the following command:
```bash
mmlearn_run 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline job_type=eval +datasets@datasets.test=ROCO datasets.test.split=test +datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=HFCLIPTokenizer +datasets/transforms@datasets.test.transform=med_clip_vision_transform datasets.test.transform.job_type=eval dataloader.test.batch_size=32 dataloader.test.num_workers=2 resume_from_checkpoint=/path/to/your/checkpoint experiment_name=test
```
