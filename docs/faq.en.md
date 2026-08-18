# FAQ

## Q. My GPU (CUDA) is not recognized. How do I install a CUDA-enabled build of PyTorch? {: #cuda-pytorch }

A. PyTorch must be installed as a build that matches your CUDA environment. In particular, the PyTorch version installed by default through the package dependencies may not support CUDA in the following cases:

* When installed from PyPI with `pip install` on Windows (a CPU-only build of PyTorch is installed)
* When using newer GPUs such as the RTX 50 series (a newer PyTorch build, e.g., one built for CUDA 12.8, may be required)

In these cases, install the dependencies first, then replace only the PyTorch-related packages with the official CUDA builds.

### Installation Steps (using pip)

After installing YomiToku, install the PyTorch packages that match your CUDA version (example: CUDA 12.8).

```bash
pip install yomitoku
pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### Installation Steps (using uv)

After cloning the repository, install the dependencies.

```bash
uv sync --extra gpu
```

Then install the PyTorch packages that match your CUDA version (example: CUDA 12.8).

=== "Windows (PowerShell)"

    ```powershell
    uv pip install --upgrade `
      torch==2.7.0 `
      torchvision==0.22.0 `
      --index-url https://download.pytorch.org/whl/cu128
    ```

=== "Linux"

    ```bash
    uv pip install --upgrade \
      torch==2.7.0 \
      torchvision==0.22.0 \
      --index-url https://download.pytorch.org/whl/cu128
    ```

!!! warning
    Re-running `uv sync` afterwards will revert PyTorch to the version declared in `pyproject.toml`. In that case, perform the replacement step again.

### Verifying CUDA Availability

After installation, verify that CUDA is recognized with the following command.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

You can confirm that inference is running on the GPU using the following methods:

* Monitor GPU utilization and memory usage with `nvidia-smi` during execution
* Check CUDA memory allocation with `torch.cuda.memory_allocated()` and similar APIs

!!! note
    If `device="cuda"` is specified while CUDA is unavailable, YomiToku logs the warning `CUDA is not available. Use CPU instead.` and falls back to running on the CPU. If processing is slower than expected, check the logs for this warning.

### Notes

* Use the latest official NVIDIA driver compatible with your CUDA version and GPU. YomiToku does not pin a specific driver version.
* YomiToku uses only the standard public APIs of PyTorch / torchvision, and this kind of minor version update is generally compatible. However, since this configuration differs from the versions declared in the package dependencies, we recommend validating with your actual documents and PDFs before production use.
