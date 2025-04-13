# MCP

This section explains how to use the Yomitoku MCP server in conjunction with Claude Desktop.

## Installing Yomitoku

First, install Yomitoku by following the "Installation with uv" section in [Installation](installation.en.md).

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
        "(Absolute path of the directory where Yomitoku was cloned)",
        "run",
        "demo/run_mcp_yomitoku.py"
      ],
      "env": {
        "RESOURCE_DIR": "(Absolute path of the directory containing files for OCR)"
      }
    }
  }
}
```

For example, if you executed `git clone https://github.com/kotaro-kinoshita/yomitoku.git` in `/Users/your-username/workspace`, then `(Directory where Yomitoku was cloned)` would be `/Users/your-username/workspace/yomitoku`, and if you use `sample.pdf` in the `yomitoku/demo` directory, specify `(Directory containing files for OCR)` as `/Users/your-username/workspace/yomitoku/demo`.

## Using Claude Desktop

* Please restart Claude Desktop to apply changes to the configuration file.

For example, if you use `yomitoku/demo/sample.pdf` as a sample, instruct as follows:

```txt
Analyze sample.pdf using OCR and translate it into English.
```
