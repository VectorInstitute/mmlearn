# Installation

!!! tip "Prerequisites"
    For local development, it is generally recommended to install *mmlearn* in a non-global environment (e.g. venv or conda).
    This will allow you to use different versions of *mmlearn* for different projects.

    You can create a virtual environment using venv with the following command:
    ```bash
    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
    ```

## Installing from PyPI

mmlearn is published on the [Python Package Index](https://pypi.org/project/mmlearn/) and can be installed using pip.

Run the following command to install the library:

```bash
python3 -m pip install mmlearn
```

!!! note
    `mmlearn` has several optional dependencies that are used for specific functionality.
    For example, the [peft](https://huggingface.co/docs/peft/index) library for parameter-efficient finetuning.
    Hence, `peft` can be installed using:

    ```bash
    python3 -m pip install mmlearn[peft]
    ```

    Specific sets of dependencies are listed below.

    | Dependency | pip extra | Notes |
    |------------|-----------|-------|
    | torchvision, timm, opencv-python | vision | Allows use of computer vision models and image processing functionality |
    | torchaudio | audio | Allows use of audio processing and audio model functionality |
    | peft | peft | Allows use of parameter-efficient fine-tuning methods |

## Installing from source

You can install mmlearn directly from a clone of the Git repository.
This can be done either by cloning the repo and installing from the local clone,
or simply installing directly via git.

```bash
git clone https://github.com/VectorInstitute/mmlearn.git
cd mmlearn
python3 -m pip install -e .
```

```bash
pip install git+https://github.com/VectorInstitute/mmlearn.git
```
