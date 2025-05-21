import csv
import io
import json
import os
from argparse import ArgumentParser
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image, load_pdf
from yomitoku.export import (
    convert_csv,
    convert_html,
    convert_json,
    convert_markdown,
)

try:
    RESOURCE_DIR = os.environ["RESOURCE_DIR"]
except KeyError:
    raise ValueError("Environment variable 'RESOURCE_DIR' is not set.")


analyzer = None


async def load_analyzer(ctx: Context) -> DocumentAnalyzer:
    """
    Load the DocumentAnalyzer instance if not already loaded.

    Args:
        ctx (Context): The context in which the analyzer is being loaded.

    Returns:
        DocumentAnalyzer: The loaded document analyzer instance.
    """
    global analyzer
    if analyzer is None:
        await ctx.info("Load document analyzer")
        analyzer = DocumentAnalyzer(visualize=False, device="cuda")
    return analyzer


mcp = FastMCP("yomitoku")


@mcp.tool()
async def process_ocr(ctx: Context, filename: str, output_format: str) -> str:
    """
    Perform OCR on the specified file in the resource direcory and convert
    the results to the desired format.

    Args:
        ctx (Context): The context in which the OCR processing is executed.
        filename (str): The name of the file to process in the resource directory.
        output_format (str): The desired format for the output. The available options are:
            - json: Outputs the text as structured data along with positional information.
            - markdown: Outputs texts and tables in Markdown format.
            - html: Outputs texts and tables in HTML format.
            - csv: Outputs texts and tables in CSV format.

    Returns:
        str: The OCR results converted to the specified format.
    """
    analyzer = await load_analyzer(ctx)

    await ctx.info("Start ocr processing")

    file_path = os.path.join(RESOURCE_DIR, filename)
    if Path(file_path).suffix[1:].lower() in ["pdf"]:
        imgs = load_pdf(file_path)
    else:
        imgs = load_image(file_path)

    results = []
    for page, img in enumerate(imgs):
        analyzer.img = img
        result, _, _ = await analyzer.run(img)
        results.append(result)
        await ctx.report_progress(page + 1, len(imgs))

    if output_format == "json":
        return json.dumps(
            [
                convert_json(
                    result,
                    out_path=None,
                    ignore_line_break=True,
                    img=img,
                    export_figure=False,
                    figure_dir=None,
                ).model_dump()
                for img, result in zip(imgs, results)
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ": "),
        )
    elif output_format == "markdown":
        return "\n".join(
            [
                convert_markdown(
                    result,
                    out_path=None,
                    ignore_line_break=True,
                    img=img,
                    export_figure=False,
                )[0]
                for img, result in zip(imgs, results)
            ]
        )
    elif output_format == "html":
        return "\n".join(
            [
                convert_html(
                    result,
                    out_path=None,
                    ignore_line_break=True,
                    img=img,
                    export_figure=False,
                    export_figure_letter="",
                )[0]
                for img, result in zip(imgs, results)
            ]
        )
    elif output_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        for img, result in zip(imgs, results):
            elements = convert_csv(
                result,
                out_path=None,
                ignore_line_break=True,
                img=img,
                export_figure=False,
            )
            for element in elements:
                if element["type"] == "table":
                    writer.writerows(element["element"])
                else:
                    writer.writerow([element["element"]])
                writer.writerow([""])
        return output.getvalue()
    else:
        raise ValueError(
            f"Unsupported output format: {output_format}."
            " Supported formats are json, markdown, html or csv."
        )


@mcp.resource("file://list")
async def get_file_list() -> list[str]:
    """
    Retrieve a list of files in the resource directory.

    Returns:
        list[str]: A list of filenames in the resource directory.
    """
    return os.listdir(RESOURCE_DIR)


def run_mcp_server(transport="stdio"):
    """
    Run the MCP server.
    """
    mcp.run(transport=transport)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the MCP server.")
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="Transport method for the MCP server.",
    )
    args = parser.parse_args()
    run_mcp_server(transport=args.transport)
