# Installation


This package requires Python 3.9 or later and PyTorch 2.5 or later for execution. PyTorch must be installed according to your CUDA version. A GPU with more than 8GB of VRAM is recommended. While it can run on a CPU, please note that the processing is not currently optimized for CPUs, which may result in longer execution times.

## from PYPI

```bash
pip install yomitoku
```

## using uv
This repository uses the package management tool [uv](https://docs.astral.sh/uv/). After installing uv, clone the repository and execute the following commands:

```bash
uv sync
```

When using uv, you need to modify the following part of the pyproject.toml file to match your CUDA version. By default, PyTorch compatible with CUDA 12.4 will be downloaded.

```pyproject.tom
[[tool.uv.index]]
name = "pytorch-cuda124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```


## using docker

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
