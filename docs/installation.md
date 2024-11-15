# Installation

本パッケージは Python3.9+, Pytorch が実行に必要です。Pytorch はご自身の環境に合わせて、インストールが必要です。計算機は GPU(> VRAM 8G)を推奨しています。CPU でも動作しますが、現在、CPU 向けに処理が最適化されておらず、実行に時間がかかりますのでご注意ください。

Linux および Mac OS で動作します。Windows 向けは動作確認未実施ですので、リポジトリの Dockerfile の利用を推奨します。

## 依存パッケージ

pdf ファイルの解析を行うためには、別途、[poppler](https://poppler.freedesktop.org/)のインストールが必要です。

=== "Mac"

```
brew install poppler
```

=== "Linux"

```
apt install poppler-utils -y
```

## TestPYPI からインストール

```bash
pip install --index-url https://test.pypi.org/simple/ yomitoku
```

## UV でのインストール

本リポジトリはパッケージ管理ツールに [UV](https://docs.astral.sh/uv/) を使用しています。UV をインストール後、リポジトリをクローンし、以下のコマンドを実行してください

```bash
uv sync
```

## Docker 環境での実行

リポジトリの直下に dockerfile を配置していますので、そちらも活用いただけます。

```bash
docker build -t yomitoku .
```

=== "Use GPU"

```bash
docker run -it --gpus all -v $(pwd):/workspace --name yomitoku yomitoku /bin/bas
```

=== "Not Use GPU"

```bash
docker run -it -v $(pwd):/workspace --name yomitoku yomitoku /bin/bas
```
