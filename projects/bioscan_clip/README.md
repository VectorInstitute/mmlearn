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

| Taxonomy | Micro top-1 accuracy (Seen) | Micro top-1 accuracy (Unseen) | Macro top-1 accuracy (Seen) | Macro top-1 accuracy (Unseen) |
|---|---|---|---|---|
| Order | 98.7 / 99.4 (+0.7) | 97.6 / 98.3 (+0.7) | 98.3 / 92.6 (-5.7) | 58.8 / 69.1 (+10.3) |
| Family | 84.6 / 89.9 (+5.3) | 79.0 / 81.5 (+2.5) | 56.3 / 76.5 (+20.2) | 35.2 / 40.3 (+5.1) |
| Genus | 58.5 / 68.4 (+9.9) | 43.5 / 48.6 (+5.1) | 30.1 / 45.6 (+15.5) | 11.7 / 15.7 (+4.0) |
| Species | 42.0 / 50.1 (+8.1) | 30.1 / 28.2 (-1.9) | 17.4 / 29.5 (+12.1) | 3.9 / 5.2 (+1.3) |

We ran the experiment with 5 different random seeds (0, 42, 1337, 1 and 1234). The results in the table are in the format
`original results / average of our results (difference between ours and the original)`. Note that our results are an *average*
of 5 runs.
</details>
