# Installation


This package requires Python 3.10 or later and PyTorch 2.5 or later for execution. PyTorch must be installed according to your CUDA version. In normal mode, the model is optimized for a GPU, and a GPU with at least 8GB of VRAM is recommended. While it can run on a CPU, expect long execution times. In efficient mode, it is designed to provide fast inference even on a CPU.

## from PYPI

```bash
pip install yomitoku
```

## using uv

This repository uses the package management tool [uv](https://docs.astral.sh/uv/). After installing uv, clone the repository and execute the following commands:

```bash
uv sync
```

Inferencing with ONNX Runtime on a GPU

```bash
uv sync --extra gpu
```

When using uv, you need to modify the following part of the pyproject.toml file to match your CUDA version. By default, PyTorch compatible with CUDA 12.4 will be downloaded.

```pyproject.tom
[[tool.uv.index]]
name = "pytorch-cuda124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

## Using docker

A Dockerfile is provided in the root of the repository, which you are welcome to use.

```bash
docker build -t yomitoku .
```

=== "GPU"
    ```bash
    docker run -it --gpus all -v $(pwd):/workspace --name yomitoku yomitoku /bin/bash
    ```

=== "CPU"
    ```bash
    docker run -it -v $(pwd):/workspace --name yomitoku yomitoku /bin/bash
    ```

## Using YomiToku in Offline Environments

YomiToku automatically downloads the model from Hugging Face Hub on its first run.
An internet connection is required at that time but, by manually pre-downloading the model using the following command, you can prepare YomiToku to run in environments without internet access:

```bash
download_model
```

By placing the downloaded repository folder `KotaroKinoshita` in the current directory at runtime, the local repository model will be loaded and executed without any internet connection.
