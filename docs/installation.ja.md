# Installation

本パッケージは Python3.10+, PyTorch が実行に必要です。PyTorch はご自身の環境に合わせて、インストールが必要です。通常モデルは GPU 向けに最適化されており、デバイスは GPU(> VRAM 8G)を推奨しています。CPU でも動作しますが、実行に時間がかかりますのでご注意ください。軽量モデルは CPU でも高速に推論できます。

## PYPI からインストール

```bash
pip install yomitoku
```

## uv でのインストール

本リポジトリはパッケージ管理ツールに [uv](https://docs.astral.sh/uv/) を使用しています。uv をインストール後、リポジトリをクローンし、以下のコマンドを実行してください

```bash
uv sync
```

ONNX Runtime を用いて GPU で推論する場合

```bash
uv sync --extra gpu
```

uvを利用する場合、`pyproject.toml`の以下の部分をご自身のcudaのバージョンに合わせて修正する必要があります。デフォルトではCUDA12.4に対応したPyTorchがダウンロードされます。

```pyproject.tom
[[tool.uv.index]]
name = "pytorch-cuda124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

## Docker 環境での実行

リポジトリの直下に dockerfile を配置していますので、そちらも活用いただけます。

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

## インターネットに接続できない環境での利用

YomiToku は初回の実行時に Hugging Face Hub からモデルを自動でダウンロードします。
その際にインターネット環境が必要ですが、事前に以下のコマンドを実行して手動でダウンロードすることでインターネットに接続できない環境でも実行することが可能です。

```bash
download_model
```

実行時にダウンロードされたリポジトリの `KotaroKinoshita` ディレクトリをカレントディレクトリに配置することで、インターネットへの接続なしに、ローカルリポジトリのモデルが呼び出され実行されます。
