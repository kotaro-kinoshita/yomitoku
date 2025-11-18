# MCP

ここではYomiTokuのMCPサーバーをClaude Desktopに連携して利用する方法を説明します。

## YomiTokuのインストール

まずは
[Installation](installation.ja.md)の「uvでのインストール」に従ってYomiTokuをインストールしてください。

ただし、`mcp`を依存関係に追加するためにインストール時には下記のように`--extra`に`mcp`を加えます。

```bash
uv sync --extra mcp
```


## Claude Desktopの設定

次にClaude Desktopの設定ファイルの`mcpServers`に以下ように設定を追加します。(設定ファイルの開き方は[こちら](https://modelcontextprotocol.io/quickstart/user)を参照してください)

```json
{
  "mcpServers": {
    "yomitoku": {
      "command": "uv",
      "args": [
        "--directory",
        "(YomiTokuをCloneしたディレクトリの絶対パス)",
        "run",
        "yomitoku_mcp"
      ],
      "env": {
        "RESOURCE_DIR": "(OCR対象のファイルがあるディレクトリの絶対パス)"
      }
    }
  }
}
```


例えば、`/Users/your-username/workspace`で`git clone https://github.com/kotaro-kinoshita/yomitoku.git`を実行した場合は、`(YomiTokuをCloneしたディレクトリ)`は`/Users/your-username/workspace/yomitoku`となり、`yomitoku/demo`ディレクトリの`sample.pdf`を用いる場合は`(OCR対象のファイルがあるディレクトリ)`を`/Users/your-username/workspace/yomitoku/demo`と指定します。

## Claude Desktopでの利用

※ 設定ファイルの変更を反映するにはClaude Desktopを再起動してください。

例えば`yomitoku/demo/sample.pdf`をサンプルとして用いる場合、下記のように指示してください。

```txt
sample.pdfをOCRで解析して要約してください。
```

## SSEサーバーの起動
環境変数の`RESOURCE_DIR`にOCRの対象画像が含まれたフォルダのパスを設定してください。
```
export RESOURCE_DIR="path of dataset"
```

以下のコマンドでSSEサーバーを起動します。
```
uv run yomitoku_mcp -t sse
```

` http://127.0.0.1:8000/sse`がSSEサーバーのエンドポイントになります。