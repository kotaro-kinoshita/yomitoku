# MCP

This section explains how to use the YomiToku MCP server in conjunction with Claude Desktop.

## Installing YomiToku

First, install YomiToku by following the "Installation with uv" section in [Installation](installation.en.md).

However, to add `mcp` as a dependency during installation, include `mcp` in `--extra` as shown below.

```bash
uv sync --extra mcp
```

## Setting up Claude Desktop

Next, add the following configuration to the `mcpServers` section of the Claude Desktop configuration file. (Refer to [here](https://modelcontextprotocol.io/quickstart/user) for how to open the configuration file)

```json
{
  "mcpServers": {
    "yomitoku": {
      "command": "uv",
      "args": [
        "--directory",
        "(Absolute path of the directory where YomiToku was cloned)",
        "run",
        "yomitoku_mcp"
      ],
      "env": {
        "RESOURCE_DIR": "(Absolute path of the directory containing files for OCR)"
      }
    }
  }
}
```

For example, if you executed `git clone https://github.com/kotaro-kinoshita/yomitoku.git` in `/Users/your-username/workspace`, then `(Directory where YomiToku was cloned)` would be `/Users/your-username/workspace/yomitoku`, and if you use `sample.pdf` in the `yomitoku/demo` directory, specify `(Directory containing files for OCR)` as `/Users/your-username/workspace/yomitoku/demo`.

## Using Claude Desktop

* Please restart Claude Desktop to apply changes to the configuration file.

For example, if you use `yomitoku/demo/sample.pdf` as a sample, instruct as follows:

```txt
Analyze sample.pdf using OCR and translate it into English.
```

## Starting the SSE Server

Set the path to the folder containing the images to be processed by OCR in the resource directory.

```bash
export RESOURCE_DIR="path of dataset"
```

Start the SSE server using the following command:

```bash
uv run yomitoku_mcp -t sse
```

The SSE server endpoint will be available at `http://127.0.0.1:8000/sse`.
