"""Regression tests for OCR aggregation without loading model weights."""

from types import SimpleNamespace

import pytest

from yomitoku.schemas.document_analyzer import Element
from yomitoku.table_semantic_parser import TableSemanticParser


def make_element(element_id, box, role=None):
    return Element(id=element_id, box=box, role=role, score=1.0, contents="stale")


def make_word(text, x, y):
    return SimpleNamespace(
        content=text,
        points=[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]],
        direction="horizontal",
    )


@pytest.mark.parametrize(
    "ids", [(None, None, None), ("same", "same", "same"), ("p0", "p1", "p2")]
)
def test_aggregate_keeps_each_elements_words_separate(ids):
    elements = [
        make_element(ids[0], [0, 0, 100, 30]),
        make_element(ids[1], [0, 50, 100, 80]),
        make_element(ids[2], [0, 100, 100, 130]),
    ]
    # Reverse input order to check reading order within a paragraph as well.
    ocr = SimpleNamespace(
        words=[
            make_word("期限", 30, 55),
            make_word("請求日", 5, 5),
            make_word("支払", 5, 55),
            make_word("領域外", 200, 200),
        ]
    )
    parser = TableSemanticParser.__new__(TableSemanticParser)

    parser.aggregate(ocr, elements)

    assert [element.contents for element in elements] == ["請求日", "支払期限", ""]
    assert [element.id for element in elements] == list(ids)


def test_aggregate_does_not_copy_text_into_group_with_same_id():
    group = make_element(None, [0, 0, 100, 100], role="group")
    paragraph = make_element(None, [0, 0, 100, 30])
    parser = TableSemanticParser.__new__(TableSemanticParser)

    parser.aggregate(
        SimpleNamespace(words=[make_word("請求日", 5, 5)]), [group, paragraph]
    )

    assert group.contents == ""
    assert paragraph.contents == "請求日"
