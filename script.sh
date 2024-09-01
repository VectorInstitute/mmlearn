#!/bin/bash

#SBATCH --job-name=mmlearn
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

PY_ARGS=${@:1}

# load virtual environment

source ~/.bashrc
source /h/negin/mmlearn/mmlearn_negin/bin/activate
export HAM10000_ROOT_DIR=/projects/multimodal/datasets/skin_cancer/

export NCCL_IB_DISABLE=1  # disable InfiniBand (the Vector cluster does not have it)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export NCCL_ASYNC_ERROR_HANDLING=1 # set to 1 for NCCL backend
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1


export MASTER_ADDR=$(hostname --fqdn)
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

export PYTHONPATH="."
nvidia-smi

# “srun” executes the script <ntasks-per-node * nodes> times
# srun --export=ALL -N $SLURM_JOB_NUM_NODES python /h/negin/mmlearn/mmlearn/cli/run.py task="ZeroShotCrossModalRetrieval" experiment="baseline" experiment_name="test" trainer.num_nodes=${SLURM_JOB_NUM_NODES} ${PY_ARGS}
# mmlearn_run --multirun hydra.launcher.mem_gb=15 hydra.launcher.partition=a40 hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 experiment=test
mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=roco job_type=eval +datasets/tokenizers@datasets.test.tokenizer=HFCLIPTokenizer +datasets/transforms@datasets.test.transform=med_clip_vision_transform datasets.test.transform.job_type=eval dataloader.test.batch_size=32 dataloader.test.num_workers=4 strict_loading=False datasets.test.split=test datasets@datasets.test=ROCO



mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=roco job_type=eval +datasets/tokenizers@datasets.test.tokenizer=HFCLIPTokenizer +datasets/transforms@datasets.test.transform=med_clip_vision_transform datasets.test.transform.job_type=eval dataloader.test.batch_size=32 dataloader.test.num_workers=4 strict_loading=False datasets.test.split=test datasets@datasets.test=ROCO
mmlearn_run --multirun hydra.launcher.mem_gb=16 hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=roco job_type=eval +datasets/tokenizers@datasets.test.tokenizer=HFCLIPTokenizer +datasets/transforms@datasets.test.transform=med_clip_vision_transform datasets.test.transform.job_type=eval dataloader.test.batch_size=16 dataloader.test.num_workers=2 strict_loading=False datasets@datasets.test=HAM10000 +datasets/tokenizers@task.evaluation_tasks.classification.task.tokenizer=HFCLIPTokenizer
# python /h/negin/mmlearn/outputs/temp.py



mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.gres=gpu:4 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=4 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=60 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' +experiment=baseline experiment_name=roco job_type=eval +datasets/tokenizers@datasets.test.tokenizer=HFCLIPTokenizer +datasets/transforms@datasets.test.transform=med_clip_vision_transform datasets.test.transform.job_type=eval dataloader.test.batch_size=32 dataloader.test.num_workers=4 strict_loading=False datasets@datasets.test=HAM10000