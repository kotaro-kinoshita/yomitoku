from pathlib import Path
import os

from mcp.server.fastmcp import Context, FastMCP

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image, load_pdf
from yomitoku.export import convert_markdown


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
async def process_ocr(filename: str, ctx: Context) -> str:
    """
    Process OCR on the given file and convert the results to markdown.

    Args:
        filename (str): The name of the file to process.
        ctx (Context): The context in which the OCR processing is executed.

    Returns:
        str: The OCR results converted to markdown format.
    """
    analyzer = await load_analyzer(ctx)

    await ctx.info("Start ocr processing")

    file_path = os.path.join(RESOURCE_DIR, filename)
    if Path(file_path).suffix[1:].lower() in ["pdf"]:
        imgs = load_pdf(file_path)
    else:
        imgs = load_image(file_path)

    markdowns = []
    for page, img in enumerate(imgs):
        analyzer.img = img
        result, _, _ = await analyzer.run(img)
        md, _ = convert_markdown(
            result,
            out_path=None,
            img=img,
            ignore_line_break=True,
            export_figure=False,
        )
        markdowns.append(md)
        await ctx.report_progress(page + 1, len(imgs))

    return "\n".join(markdowns)


@mcp.resource("file://list")
async def get_file_list() -> list[str]:
    """
    Retrieve a list of files in the resource directory.

    Returns:
        list[str]: A list of filenames in the resource directory.
    """
    return os.listdir(RESOURCE_DIR)


if __name__ == "__main__":
    mcp.run(transport="stdio")
