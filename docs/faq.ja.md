# FAQ

## Q. GPU（CUDA）が認識されません。CUDA に対応した PyTorch のインストール方法を教えてください。 {: #cuda-pytorch }

A. PyTorch はご自身の CUDA 環境に合わせたビルドをインストールする必要があります。特に以下のケースでは、依存関係として標準でインストールされる PyTorch では CUDA が利用できないことがあります。

* Windows で PyPI から `pip install` した場合（CPU 版の PyTorch がインストールされます）
* RTX 50 シリーズなどの新しい GPU を利用する場合（CUDA 12.8 対応ビルドなど、より新しい PyTorch が必要になることがあります）

この場合、先に依存パッケージを導入した後、PyTorch 関連パッケージのみを公式の CUDA ビルドへ置き換えてください。

### 導入手順（pip を利用する場合）

YomiToku をインストール後、CUDA のバージョンに対応する PyTorch 関連パッケージをインストールします（例: CUDA 12.8）。

```bash
pip install yomitoku
pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 導入手順（uv を利用する場合）

リポジトリをクローン後、依存関係をインストールします。

```bash
uv sync --extra gpu
```

続いて、CUDA のバージョンに対応する PyTorch 関連パッケージをインストールします（例: CUDA 12.8）。

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
    この後に `uv sync` を再実行すると、PyTorch は `pyproject.toml` の宣言バージョンへ戻ります。その場合は置き換え手順を再度実施してください。

### CUDA 認識の確認

導入後、以下のコマンドで CUDA が認識されていることを確認してください。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

推論が GPU 上で実行されていることは、以下の方法で確認できます。

* 実行中に `nvidia-smi` で GPU 使用率・メモリ使用量を確認する
* `torch.cuda.memory_allocated()` 等で CUDA メモリの確保状況を確認する

!!! note
    CUDA が利用できない状態で `device="cuda"` を指定した場合、`CUDA is not available. Use CPU instead.` という警告を出力したうえで CPU にフォールバックして実行されます。処理が想定より遅い場合は、ログに本警告が出ていないか確認してください。

### 注意事項

* NVIDIA ドライバは、利用する CUDA バージョンおよび GPU に対応した公式最新版をご利用ください。YomiToku 側で特定バージョンは固定していません。
* YomiToku は PyTorch / torchvision の一般的な公開 API のみを利用しており、この種のマイナーバージョン更新は通常互換性があります。ただし、依存関係として宣言しているバージョンとは異なる組み合わせとなるため、本番利用の前には実際の文書・PDF での事前検証を推奨します。
