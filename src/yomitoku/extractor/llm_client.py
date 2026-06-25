import json
import re
from typing import Any, Dict, List

from ..utils.logger import set_logger

logger = set_logger(__name__, "INFO")


def _parse_json_response(text: str) -> Any:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for LLM extraction. "
            "Install it with: pip install yomitoku[extract]"
        )

    client = OpenAI(base_url=api_base, api_key=api_key or "EMPTY")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.info(
            "response_format=json_object not supported, falling back to plain text"
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    content = response.choices[0].message.content
    return _parse_json_response(content)
