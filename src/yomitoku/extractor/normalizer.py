import re
from typing import Callable, Dict, Optional

import jaconv


_REGISTRY: Dict[str, Callable[[str], str]] = {}


def register(name: str):
    def decorator(fn: Callable[[str], str]):
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_normalizer(name: str) -> Optional[Callable[[str], str]]:
    return _REGISTRY.get(name)


def apply_normalize(text, rule: Optional[str]) -> str:
    if rule is None:
        return text if isinstance(text, str) else str(text)
    if not isinstance(text, str):
        text = str(text)
    fn = get_normalizer(rule)
    if fn is None:
        return text
    return fn(text)


@register("strip_spaces")
def strip_spaces(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text)


@register("numeric")
def numeric(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)
    text = text.replace(",", "").replace("\u3001", "")
    text = re.sub(r"[^\d.\-+]", "", text)
    return text


@register("phone_jp")
def phone_jp(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits


_ERA_KANJI_MAP = {
    "\u4ee4\u548c": 2018,
    "\u5e73\u6210": 1988,
    "\u662d\u548c": 1925,
    "\u5927\u6b63": 1911,
    "\u660e\u6cbb": 1867,
}

_ERA_ABBREV_MAP = {
    "R": 2018,
    "H": 1988,
    "S": 1925,
    "T": 1911,
    "M": 1867,
}


def _parse_date(text: str):
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)

    for era, offset in _ERA_KANJI_MAP.items():
        m = re.search(
            rf"{era}\s*(\d+)\s*\u5e74\s*(\d+)\s*\u6708\s*(\d+)\s*\u65e5", text
        )
        if m:
            return offset + int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.search(
        r"([RHSTM])\s*(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{1,2})", text
    )
    if m:
        offset = _ERA_ABBREV_MAP.get(m.group(1))
        if offset:
            return offset + int(m.group(2)), int(m.group(3)), int(m.group(4))

    m = re.search(
        r"([RHSTM])\s*(\d{1,2})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5",
        text,
    )
    if m:
        offset = _ERA_ABBREV_MAP.get(m.group(1))
        if offset:
            return offset + int(m.group(2)), int(m.group(3)), int(m.group(4))

    m = re.search(r"(\d{4})\s*\u5e74\s*(\d{1,2})\s*\u6708\s*(\d{1,2})\s*\u65e5", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.search(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    return None


@register("date_jp")
def date_jp(text: str) -> str:
    result = _parse_date(text)
    if result:
        year, month, day = result
        return f"{year:04d}-{month:02d}-{day:02d}"
    return text


@register("date_yyyymmdd")
def date_yyyymmdd(text: str) -> str:
    result = _parse_date(text)
    if result:
        year, month, day = result
        return f"{year:04d}{month:02d}{day:02d}"
    return text


@register("time_jp")
def time_jp(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)

    m = re.search(r"(\d{1,2})\s*\u6642\s*(\d{1,2})\s*\u5206\s*(\d{1,2})\s*\u79d2", text)
    if m:
        return f"{int(m.group(1))}\u6642{int(m.group(2)):02d}\u5206{int(m.group(3)):02d}\u79d2"

    m = re.search(r"(\d{1,2})\s*\u6642\s*(\d{1,2})\s*\u5206", text)
    if m:
        return f"{int(m.group(1))}\u6642{int(m.group(2)):02d}\u5206"

    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        return f"{int(m.group(1))}\u6642{int(m.group(2)):02d}\u5206{int(m.group(3)):02d}\u79d2"

    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        return f"{int(m.group(1))}\u6642{int(m.group(2)):02d}\u5206"

    return text


@register("time_hms")
def time_hms(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)

    m = re.search(r"(\d{1,2})\s*\u6642\s*(\d{1,2})\s*\u5206\s*(\d{1,2})\s*\u79d2", text)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3)):02d}"

    m = re.search(r"(\d{1,2})\s*\u6642\s*(\d{1,2})\s*\u5206", text)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:00"

    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3)):02d}"

    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:00"

    return text


@register("alphanumeric")
def alphanumeric(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)
    return re.sub(r"[^a-zA-Z0-9]", "", text)


@register("hiragana")
def hiragana(text: str) -> str:
    text = jaconv.kata2hira(text)
    return re.sub(r"[^\u3040-\u309F]", "", text)


@register("katakana")
def katakana(text: str) -> str:
    text = jaconv.hira2kata(text)
    return re.sub(r"[^\u30A0-\u30FF]", "", text)


@register("postal_code_jp")
def postal_code_jp(text: str) -> str:
    text = jaconv.z2h(text, digit=True, ascii=True, kana=False)
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"
    return digits
