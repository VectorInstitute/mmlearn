## BIOSCAN-CLIP
BIOSCAN-CLIP uses contrastive learning to align images, DNA barcodes and a text in a common embedding space. This project
aims at reproducing the results of the [BIOSCAN-CLIP paper](https://arxiv.org/pdf/2405.17537). You can find much more
details about the project in the paper and the [official repository](https://github.com/3dlg-hcvc/bioscan-clip).

## Setup
To get started, it is recommended that you create a virtual environment for the project. You can do this by running the
following commands:
```bash
python -m venv bioscan_clip
source bioscan_clip/bin/activate
```

Then, you can install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```
**NOTE**: It is assumed that the requirements for the `mmlearn` package have already been installed in a virtual environment.
If not, please refer to the README file in the `mmlearn` package for installation instructions.

### Downloading the dataset and pretrained checkpoints
The BIOSCAN-CLIP model is trained on the BIOSCAN-1M and BIOSCAN-5M datasets.

#### Downloading the BIOSCAN-1M dataset
To download the BIOSCAN-1M dataset, you can run the following command:
```bash
mkdir -p data/BIOSCAN_1M && cd data/BIOSCAN_1M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/data/version_0.2.1/BioScan_data_in_splits.hdf5
```
**NOTE**: Add the path to the `BIOSCAN_1M` dataset to the environment variable `BIOSCAN_1M_HDF5`.

#### Downloading the BIOSCAN-5M dataset
To download the BIOSCAN-5M dataset, you can run the following command:
```bash
mkdir -p data/BIOSCAN_5M && cd data/BIOSCAN_5M
wget https://aspis.cmpt.sfu.ca/projects/bioscan/BIOSCAN_CLIP_for_downloading/BIOSCAN_5M.hdf5
```
**NOTE**: Add the path to the `BIOSCAN_5M` dataset to the environment variable `BIOSCAN_5M_HDF5`.

#### Downloading the pretrained checkpoints for the BarcodeBERT model
To download the pretrained checkpoints for the BarcodeBERT model, run the following command:
```bash
mkdir -p ckpt/barcode_bert && cd ckpt/barcode_bert
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/BarcodeBERT/model_41.pth
```
**NOTE**: Add the path to the pretrained checkpoint for the BarcodeBERT model to the environment variable `BARCODEBERT_5MER`.


## Running Experiments
### Pretraining
To start a pretraining run with the BIOSCAN-1M dataset, use the following command:

```bash
mmlearn_run --multirun hydra.launcher.mem_gb=32 hydra.launcher.qos=<enter_your_qos_here> hydra.launcher.partition=a40 hydra.launcher.gres=gpu:2 hydra.launcher.cpus_per_task=8 hydra.launcher.tasks_per_node=2 hydra.launcher.nodes=1 hydra.launcher.stderr_to_stdout=true hydra.launcher.timeout_min=1440 '+hydra.launcher.additional_parameters={export: ALL}' 'hydra.searchpath=[pkg://projects.bioscan_clip.configs]' +experiment=bioscan_1m experiment_name=bioscan1m_rgb_text_dna_test
```
**NOTE**: Replace `<enter_your_qos_here>` with the appropriate QOS for your cluster. To run the experiment outside a SLURM
environment, remove the `hydra.launcher.*` arguments.

### Evaluation
To run taxonomic classification evaluation task on the BIOSCAN-1M dataset, use the following command:
```bash
mmlearn_run 'hydra.searchpath=[pkg://projects.bioscan_clip.configs]' +experiment=bioscan_1m experiment_name=bioscan1m_eval job_type=eval resume_from_checkpoint=<path_to_checkpoint> strict_loading=false trainer.devices=1
```

<details>
<summary><b>Results</b></summary>
Here are the results we obtained from running the pretraining with the `bioscan_1m.yaml` configuration file:

| Taxonomy | Macro top-1 accuracy (Seen) | Macro top-1 accuracy (Unseen) | Micro top-1 accuracy (Seen) | Micro top-1 accuracy (Unseen) |
|---|---|---|---|---|
| Order | 98.7 (+0.6) | 97.6 (+0.8) | 98.3 (-19.8) | 58.8 (+11.6) |
| Family | 84.6 (+4.6) | 79.0 (+2.9) | 56.3 (+18.6) | 35.2 (+4.8) |
| Genus | 58.5 (+10.3) | 43.5 (+7.8) | 30.1 (+14.9) | 11.7 (+3.6) |
| Species | 42.0 (+7.5) | 30.1 (+2.6) | 17.4 (+11.5) | 3.9 (+1.0) |

The numbers in parentheses represent the difference in accuracy between the original results and the results obtained with
the `bioscan_1m.yaml` configuration file.
</details>
