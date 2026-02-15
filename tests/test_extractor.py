# tests/test_extractor.py
"""Unit tests for yomitoku extractor modules.

Covers:
- schema.py: YAML parsing, validation
- normalizer.py: all normalization rules
- resolver.py: build_lookup, _resolve_element, resolve_fields
- rule_pipeline.py: scalar/table extraction with cell_id, bbox, description, regex
- pipeline.py: _normalize_resolved_fields, _build_output
- prompt.py: build_messages structure
"""

from __future__ import annotations

from types import SimpleNamespace


# =========================================================================
# schema.py
# =========================================================================
class TestExtractionSchema:
    def test_from_yaml_scalar_fields(self, tmp_path):
        yaml_content = """\
fields:
  - name: phone
    description: 電話番号
    type: string
    normalize: phone_jp
  - name: amount
    description: 合計金額
    type: number
    normalize: numeric
"""
        p = tmp_path / "schema.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.from_yaml(str(p))
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "phone"
        assert schema.fields[0].description == "電話番号"
        assert schema.fields[0].type == "string"
        assert schema.fields[0].normalize == "phone_jp"
        assert schema.fields[0].structure == "scalar"
        assert schema.fields[1].type == "number"

    def test_from_yaml_table_field(self, tmp_path):
        yaml_content = """\
fields:
  - name: items
    structure: table
    columns:
      - name: product
        description: 商品名
        type: string
      - name: price
        description: 金額
        type: number
        normalize: numeric
"""
        p = tmp_path / "schema.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.from_yaml(str(p))
        assert len(schema.fields) == 1
        f = schema.fields[0]
        assert f.structure == "table"
        assert len(f.columns) == 2
        assert f.columns[0].name == "product"
        assert f.columns[1].normalize == "numeric"

    def test_from_yaml_with_cell_id_bbox_regex(self, tmp_path):
        yaml_content = """\
fields:
  - name: f1
    cell_id: c12
    type: string
  - name: f2
    bbox: [100, 200, 300, 400]
    type: string
  - name: f3
    regex: 'T\\d{13}'
    type: string
"""
        p = tmp_path / "schema.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.from_yaml(str(p))
        assert schema.fields[0].cell_id == "c12"
        assert schema.fields[1].bbox == [100, 200, 300, 400]
        assert schema.fields[2].regex == r"T\d{13}"

    def test_from_yaml_structure_kv(self, tmp_path):
        yaml_content = """\
fields:
  - name: phone
    description: 電話番号
    structure: kv
    type: string
"""
        p = tmp_path / "schema.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.from_yaml(str(p))
        assert schema.fields[0].structure == "kv"

    def test_column_schema_with_cell_id_bbox(self, tmp_path):
        yaml_content = """\
fields:
  - name: tbl
    structure: table
    columns:
      - name: col1
        cell_id: c5
      - name: col2
        bbox: [10, 20, 30, 40]
      - name: col3
        description: ヘッダー
"""
        p = tmp_path / "schema.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.from_yaml(str(p))
        cols = schema.fields[0].columns
        assert cols[0].cell_id == "c5"
        assert cols[1].bbox == [10, 20, 30, 40]
        assert cols[2].description == "ヘッダー"


# =========================================================================
# normalizer.py
# =========================================================================
class TestNormalizer:
    def test_strip_spaces(self):
        from yomitoku.extractor.normalizer import strip_spaces

        assert strip_spaces("東京都　渋谷区") == "東京都渋谷区"
        assert strip_spaces("a b c") == "abc"
        assert strip_spaces("hello") == "hello"

    def test_numeric(self):
        from yomitoku.extractor.normalizer import numeric

        assert numeric("１，２３４円") == "1234"
        assert numeric("¥3,000") == "3000"
        assert numeric("-100.5") == "-100.5"

    def test_phone_jp_11_digits(self):
        from yomitoku.extractor.normalizer import phone_jp

        assert phone_jp("０９０１２３４５６７８") == "090-1234-5678"

    def test_phone_jp_10_digits(self):
        from yomitoku.extractor.normalizer import phone_jp

        assert phone_jp("０３１２３４５６７８") == "031-234-5678"

    def test_postal_code_jp(self):
        from yomitoku.extractor.normalizer import postal_code_jp

        assert postal_code_jp("１２３４５６７") == "123-4567"
        assert postal_code_jp("123-4567") == "123-4567"

    def test_date_jp_kanji_era(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("令和6年3月15日") == "2024-03-15"
        assert date_jp("平成30年1月1日") == "2018-01-01"
        assert date_jp("昭和60年12月25日") == "1985-12-25"

    def test_date_jp_abbrev_era(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("R6/3/15") == "2024-03-15"
        assert date_jp("H30.1.1") == "2018-01-01"
        assert date_jp("S60-12-25") == "1985-12-25"

    def test_date_jp_abbrev_era_kanji(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("R5年1月11日") == "2023-01-11"

    def test_date_jp_western(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("2024年3月15日") == "2024-03-15"
        assert date_jp("2024/3/15") == "2024-03-15"
        assert date_jp("2024-03-15") == "2024-03-15"

    def test_date_jp_no_match_returns_original(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("不明な日付") == "不明な日付"

    def test_date_yyyymmdd(self):
        from yomitoku.extractor.normalizer import date_yyyymmdd

        assert date_yyyymmdd("令和6年3月15日") == "20240315"
        assert date_yyyymmdd("R5/01/11") == "20230111"
        assert date_yyyymmdd("2024/3/15") == "20240315"

    def test_time_jp(self):
        from yomitoku.extractor.normalizer import time_jp

        assert time_jp("14:30") == "14時30分"
        assert time_jp("9:05:30") == "9時05分30秒"
        assert time_jp("14時30分") == "14時30分"

    def test_time_hms(self):
        from yomitoku.extractor.normalizer import time_hms

        assert time_hms("14時30分") == "14:30:00"
        assert time_hms("9:5") == "09:05:00"
        assert time_hms("14:30:59") == "14:30:59"

    def test_apply_normalize_none_rule(self):
        from yomitoku.extractor.normalizer import apply_normalize

        assert apply_normalize("hello", None) == "hello"

    def test_apply_normalize_unknown_rule(self):
        from yomitoku.extractor.normalizer import apply_normalize

        assert apply_normalize("hello", "unknown_rule") == "hello"

    def test_apply_normalize_valid_rule(self):
        from yomitoku.extractor.normalizer import apply_normalize

        assert apply_normalize("１２３", "numeric") == "123"

    def test_apply_normalize_int_value(self):
        from yomitoku.extractor.normalizer import apply_normalize

        assert apply_normalize(12345, "phone_jp") == "12345"
        assert apply_normalize(12345, None) == "12345"
        assert apply_normalize(100, "numeric") == "100"

    def test_alphanumeric(self):
        from yomitoku.extractor.normalizer import alphanumeric

        assert alphanumeric("ＡＢＣ１２３") == "ABC123"
        assert alphanumeric("abc 123!@#") == "abc123"
        assert alphanumeric("商品A01") == "A01"

    def test_hiragana(self):
        from yomitoku.extractor.normalizer import hiragana

        assert hiragana("カタカナ") == "かたかな"
        assert hiragana("ひらがな") == "ひらがな"
        assert hiragana("テスト123abc") == "てすと"
        assert hiragana("カタカナとひらがな混在") == "かたかなとひらがな"

    def test_katakana(self):
        from yomitoku.extractor.normalizer import katakana

        assert katakana("ひらがな") == "ヒラガナ"
        assert katakana("カタカナ") == "カタカナ"
        assert katakana("テスト123abc") == "テスト"
        assert katakana("ひらがなとカタカナ混在") == "ヒラガナトカタカナ"


# =========================================================================
# resolver.py
# =========================================================================
def _make_cell(cell_id, box, contents, role="cell"):
    """Create a CellSchema-like object."""
    return SimpleNamespace(id=cell_id, box=box, contents=contents, role=role, meta={})


def _make_paragraph(pid, box, contents, role=None):
    """Create an Element-like object (paragraph)."""
    return SimpleNamespace(id=pid, box=box, contents=contents, score=1.0, role=role)


def _make_word(content, points):
    """Create a WordPrediction-like object."""
    from yomitoku.schemas.document_analyzer import WordPrediction

    return WordPrediction(
        content=content,
        points=points,
        direction="horizontal",
        rec_score=1.0,
        det_score=1.0,
    )


def _make_table(cells_dict, kv_items=None, grids=None):
    """Create a minimal TableSemanticContentsSchema-like object."""
    table = SimpleNamespace(
        id="t0",
        style="border",
        box=[0, 0, 1000, 1000],
        cells=cells_dict,
        kv_items=kv_items or [],
        grids=grids or [],
    )

    # find_cell_by_id / search_cells_by_bbox stubs
    def find_cell_by_id(cell_id):
        return cells_dict.get(str(cell_id))

    def search_cells_by_bbox(box):
        from yomitoku.utils.misc import is_contained

        result = []
        for cell in cells_dict.values():
            if cell.role == "group":
                continue
            if is_contained(box, cell.box, threshold=0.5):
                result.append(cell)
        return result

    def search_cells_by_query(query):
        from yomitoku.extractor.rule_pipeline import _normalize_text

        q = _normalize_text(query)
        out = []
        for cell in cells_dict.values():
            if not cell.contents or cell.role == "group":
                continue
            if q in _normalize_text(cell.contents):
                out.append(cell)
        return out

    table.find_cell_by_id = find_cell_by_id
    table.search_cells_by_bbox = search_cells_by_bbox
    table.search_cells_by_query = search_cells_by_query

    return table


def _make_semantic_info(tables=None, paragraphs=None, words=None):
    """Create a minimal TableSemanticParserSchema-like object."""
    info = SimpleNamespace(
        tables=tables or [],
        paragraphs=paragraphs or [],
        words=words or [],
    )

    def search_kv_items_by_key(key):
        from yomitoku.extractor.rule_pipeline import _normalize_text

        q = _normalize_text(key)
        results = []
        for table in info.tables:
            for kv in table.kv_items:
                key_cells = [
                    table.cells.get(k)
                    for k in (kv.key if isinstance(kv.key, list) else [kv.key])
                ]
                key_text = "".join((c.contents or "") for c in key_cells if c)
                if q in _normalize_text(key_text):
                    value_cell = table.cells.get(kv.value)
                    results.append({"key": key_cells, "value": value_cell})
        return results

    info.search_kv_items_by_key = search_kv_items_by_key
    return info


class TestBuildLookup:
    def test_build_lookup_cells_and_paragraphs(self):
        from yomitoku.extractor.resolver import build_lookup

        cell = _make_cell("c0", [0, 0, 10, 10], "hello")
        para = _make_paragraph("p0", [20, 20, 30, 30], "world")
        word = _make_word("foo", [[0, 0], [10, 0], [10, 10], [0, 10]])

        table = SimpleNamespace(cells={"c0": cell})
        info = SimpleNamespace(tables=[table], paragraphs=[para], words=[word])

        lookup = build_lookup(info)
        assert "c0" in lookup
        assert "p0" in lookup
        assert "w0" in lookup

    def test_build_lookup_paragraph_without_id(self):
        from yomitoku.extractor.resolver import build_lookup

        para = _make_paragraph(None, [0, 0, 10, 10], "text")
        info = SimpleNamespace(tables=[], paragraphs=[para], words=[])

        lookup = build_lookup(info)
        assert "p0" in lookup


class TestResolveElement:
    def test_resolve_cell(self):
        from yomitoku.extractor.resolver import _resolve_element

        cell = _make_cell("c0", [10, 20, 30, 40], "text")
        lookup = {"c0": cell}

        elem = _resolve_element("c0", lookup)
        assert elem is not None
        assert elem.id == "c0"
        assert elem.box == [10, 20, 30, 40]
        assert elem.contents == "text"

    def test_resolve_word(self):
        from yomitoku.extractor.resolver import _resolve_element

        word = _make_word("abc", [[100, 200], [300, 200], [300, 250], [100, 250]])
        lookup = {"w0": word}

        elem = _resolve_element("w0", lookup)
        assert elem is not None
        assert elem.id == "w0"
        assert elem.contents == "abc"
        assert list(elem.box) == [100, 200, 300, 250]

    def test_resolve_missing(self):
        from yomitoku.extractor.resolver import _resolve_element

        elem = _resolve_element("missing", {})
        assert elem is None


class TestResolveFields:
    def test_resolve_scalar_field(self):
        from yomitoku.extractor.resolver import resolve_fields

        cell = _make_cell("c0", [0, 0, 10, 10], "hello")
        lookup = {"c0": cell}

        llm_results = [
            {
                "name": "field1",
                "value": "hello",
                "raw_text": "hello",
                "confidence": "high",
                "source": "kv",
                "cell_ids": ["c0"],
            }
        ]

        resolved = resolve_fields(llm_results, lookup)
        assert len(resolved) == 1
        assert resolved[0].name == "field1"
        assert resolved[0].value == "hello"
        assert len(resolved[0].elements) == 1
        assert resolved[0].elements[0].id == "c0"

    def test_resolve_table_field(self):
        from yomitoku.extractor.resolver import resolve_fields

        cell_a = _make_cell("c0", [0, 0, 10, 10], "ProductA")
        cell_b = _make_cell("c1", [10, 0, 20, 10], "100")
        lookup = {"c0": cell_a, "c1": cell_b}

        llm_results = [
            {
                "name": "items",
                "value": [
                    {
                        "product": {"value": "ProductA", "cell_ids": ["c0"]},
                        "price": {"value": "100", "cell_ids": ["c1"]},
                    }
                ],
                "raw_text": "",
                "confidence": "high",
                "source": "grid",
                "cell_ids": [],
            }
        ]

        resolved = resolve_fields(llm_results, lookup)
        assert len(resolved) == 1
        elems = resolved[0].elements
        assert len(elems) == 2
        labels = {e.label for e in elems}
        assert labels == {"product", "price"}

    def test_resolve_dict_value_extracts_nested_cell_ids(self):
        """LLMがKVフィールドをdict形式で返した場合でもelementsが解決される"""
        from yomitoku.extractor.resolver import resolve_fields

        cell = _make_cell("c6", [100, 200, 300, 250], "100031582200")
        lookup = {"c6": cell}

        llm_results = [
            {
                "name": "model_code",
                "value": {
                    "value": "100031582200",
                    "cell_ids": ["c6"],
                },
                "raw_text": "100031582200",
                "confidence": "high",
                "source": "kv",
                "cell_ids": [],
            }
        ]

        resolved = resolve_fields(llm_results, lookup)
        assert len(resolved) == 1
        assert resolved[0].value == "100031582200"
        assert len(resolved[0].elements) == 1
        assert resolved[0].elements[0].id == "c6"
        assert resolved[0].elements[0].box == [100, 200, 300, 250]

    def test_resolve_skips_non_dict_items(self):
        """LLMがresults配列に文字列等を返した場合にスキップされる"""
        from yomitoku.extractor.resolver import resolve_fields

        llm_results = [
            "unexpected string",
            {
                "name": "field1",
                "value": "hello",
                "raw_text": "hello",
                "confidence": "high",
                "source": "kv",
                "cell_ids": [],
            },
            42,
        ]

        resolved = resolve_fields(llm_results, {})
        assert len(resolved) == 1
        assert resolved[0].name == "field1"


# =========================================================================
# rule_pipeline.py - scalar extraction
# =========================================================================
class TestExtractScalarByCellId:
    def test_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_cell_id

        cell = _make_cell("c5", [0, 0, 100, 50], "東京都渋谷区")
        table = _make_table({"c5": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="address", cell_id="c5")
        result = _extract_scalar_by_cell_id(info, field_schema)

        assert result is not None
        assert result.value == "東京都渋谷区"
        assert result.source == "cell_id"
        assert result.confidence == "high"

    def test_not_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_cell_id

        table = _make_table({})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="test", cell_id="c99")
        result = _extract_scalar_by_cell_id(info, field_schema)
        assert result is None


class TestExtractScalarByBbox:
    def test_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_bbox

        cell = _make_cell("c1", [100, 100, 200, 150], "value1")
        table = _make_table({"c1": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="f1", bbox=[100, 100, 200, 150])
        result = _extract_scalar_by_bbox(info, field_schema)

        assert result is not None
        assert result.value == "value1"
        assert result.source == "bbox"

    def test_not_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_bbox

        cell = _make_cell("c1", [100, 100, 200, 150], "value1")
        table = _make_table({"c1": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="f1", bbox=[900, 900, 950, 950])
        result = _extract_scalar_by_bbox(info, field_schema)
        assert result is None


class TestExtractScalarByRegex:
    def test_match_in_cell(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_regex

        cell = _make_cell("c1", [0, 0, 100, 50], "登録番号T1234567890123です")
        table = _make_table({"c1": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="invoice", regex=r"T\d{13}")
        result = _extract_scalar_by_regex(info, field_schema)

        assert result is not None
        assert result.value == "T1234567890123"
        assert result.raw_text == "登録番号T1234567890123です"
        assert result.source == "regex"

    def test_match_in_paragraph(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_regex

        para = _make_paragraph("p0", [0, 0, 100, 50], "TEL: 03-1234-5678")
        info = _make_semantic_info(paragraphs=[para])

        field_schema = SimpleNamespace(name="phone", regex=r"\d{2,4}-\d{2,4}-\d{2,4}")
        result = _extract_scalar_by_regex(info, field_schema)

        assert result is not None
        assert result.value == "03-1234-5678"
        assert result.source == "regex"

    def test_match_in_word(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_regex

        word = _make_word("T9876543210123", [[0, 0], [100, 0], [100, 20], [0, 20]])
        info = _make_semantic_info(words=[word])

        field_schema = SimpleNamespace(name="invoice", regex=r"T\d{13}")
        result = _extract_scalar_by_regex(info, field_schema)

        assert result is not None
        assert result.value == "T9876543210123"
        assert result.elements[0].id == "w0"

    def test_skips_group_cells(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_regex

        group_cell = _make_cell("c1", [0, 0, 100, 50], "T1234567890123", role="group")
        table = _make_table({"c1": group_cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="invoice", regex=r"T\d{13}")
        result = _extract_scalar_by_regex(info, field_schema)
        assert result is None

    def test_no_match(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_by_regex

        cell = _make_cell("c1", [0, 0, 100, 50], "何もマッチしない")
        table = _make_table({"c1": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(name="invoice", regex=r"T\d{13}")
        result = _extract_scalar_by_regex(info, field_schema)
        assert result is None


class TestExtractScalarField:
    def test_priority_cell_id_first(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        cell = _make_cell("c5", [0, 0, 100, 50], "cell_value")
        table = _make_table({"c5": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="f1", cell_id="c5", bbox=None, description="", regex=None
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.value == "cell_value"
        assert result.source == "cell_id"

    def test_fallback_to_description_kv(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        key_cell = _make_cell("c1", [0, 0, 50, 30], "電話番号", role="header")
        val_cell = _make_cell("c2", [50, 0, 150, 30], "03-1234-5678")
        kv = SimpleNamespace(key=["c1"], value="c2")
        table = _make_table({"c1": key_cell, "c2": val_cell}, kv_items=[kv])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="phone", cell_id=None, bbox=None, description="電話番号", regex=None
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.value == "03-1234-5678"
        assert result.source == "kv"

    def test_fallback_to_regex(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        cell = _make_cell("c1", [0, 0, 100, 50], "番号T1234567890123")
        table = _make_table({"c1": cell})
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="invoice", cell_id=None, bbox=None, description="", regex=r"T\d{13}"
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.value == "T1234567890123"
        assert result.source == "regex"

    def test_not_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        info = _make_semantic_info()

        field_schema = SimpleNamespace(
            name="missing", cell_id=None, bbox=None, description="", regex=None
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.value == ""
        assert result.source == "not_found"
        assert result.confidence == "low"

    def test_description_cell_query_fallback(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        cell = _make_cell("c1", [0, 0, 100, 50], "電話番号: 03-1234-5678")
        table = _make_table({"c1": cell}, kv_items=[])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="phone", cell_id=None, bbox=None, description="電話番号", regex=None
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.source == "cell_query"
        assert result.confidence == "medium"
        assert "03-1234-5678" in result.value

    def test_description_paragraph_fallback(self):
        from yomitoku.extractor.rule_pipeline import _extract_scalar_field

        para = _make_paragraph("p0", [0, 0, 200, 50], "住所：東京都渋谷区1-1-1")
        info = _make_semantic_info(paragraphs=[para])

        field_schema = SimpleNamespace(
            name="address", cell_id=None, bbox=None, description="住所", regex=None
        )
        result = _extract_scalar_field(info, field_schema)
        assert result.source == "paragraph"
        assert result.confidence == "medium"
        assert "東京都渋谷区" in result.value


# =========================================================================
# rule_pipeline.py - table extraction
# =========================================================================
class TestMatchColHeader:
    def test_match_by_cell_id(self):
        from yomitoku.extractor.rule_pipeline import _match_col_header

        table = _make_table(
            {"c1": _make_cell("c1", [0, 0, 50, 30], "日付", role="header")}
        )
        col_schema = SimpleNamespace(
            name="date", cell_id="c1", bbox=None, description=""
        )

        assert _match_col_header(table, ["c1"], col_schema) is True

    def test_no_match_by_cell_id(self):
        from yomitoku.extractor.rule_pipeline import _match_col_header

        table = _make_table(
            {"c1": _make_cell("c1", [0, 0, 50, 30], "日付", role="header")}
        )
        col_schema = SimpleNamespace(
            name="date", cell_id="c99", bbox=None, description=""
        )

        assert _match_col_header(table, ["c1"], col_schema) is False

    def test_match_by_description(self):
        from yomitoku.extractor.rule_pipeline import _match_col_header

        table = _make_table(
            {"c1": _make_cell("c1", [0, 0, 50, 30], "日付", role="header")}
        )
        col_schema = SimpleNamespace(
            name="date", cell_id=None, bbox=None, description="日付"
        )

        assert _match_col_header(table, ["c1"], col_schema) is True

    def test_match_by_name_when_no_description(self):
        from yomitoku.extractor.rule_pipeline import _match_col_header

        table = _make_table(
            {"c1": _make_cell("c1", [0, 0, 50, 30], "date", role="header")}
        )
        col_schema = SimpleNamespace(
            name="date", cell_id=None, bbox=None, description=""
        )

        assert _match_col_header(table, ["c1"], col_schema) is True


class TestExtractTableField:
    def _make_grid(self, col_headers, data):
        return SimpleNamespace(col_headers=col_headers, data=data)

    def test_basic_table_extraction(self):
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        cells = {
            "h1": _make_cell("h1", [0, 0, 50, 30], "商品名", role="header"),
            "h2": _make_cell("h2", [50, 0, 100, 30], "金額", role="header"),
            "c1": _make_cell("c1", [0, 30, 50, 60], "りんご"),
            "c2": _make_cell("c2", [50, 30, 100, 60], "100"),
            "c3": _make_cell("c3", [0, 60, 50, 90], "みかん"),
            "c4": _make_cell("c4", [50, 60, 100, 90], "200"),
        }
        grid = self._make_grid(
            col_headers=[["h1"], ["h2"]],
            data=[["c1", "c2"], ["c3", "c4"]],
        )
        table = _make_table(cells, grids=[grid])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="items",
            columns=[
                SimpleNamespace(
                    name="product", cell_id=None, bbox=None, description="商品名"
                ),
                SimpleNamespace(
                    name="price", cell_id=None, bbox=None, description="金額"
                ),
            ],
        )

        result = _extract_table_field(info, field_schema)
        assert result.source == "grid"
        assert result.confidence == "high"
        assert len(result.value) == 2
        assert result.value[0]["product"]["value"] == "りんご"
        assert result.value[0]["price"]["value"] == "100"
        assert result.value[1]["product"]["value"] == "みかん"
        assert result.value[1]["price"]["value"] == "200"

    def test_table_extraction_skips_header_in_data(self):
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        cells = {
            "h1": _make_cell("h1", [0, 0, 50, 30], "名前", role="header"),
            "c1": _make_cell("c1", [0, 30, 50, 60], "太郎"),
        }
        grid = self._make_grid(
            col_headers=[["h1"]],
            data=[["h1"], ["c1"]],  # first row is header
        )
        table = _make_table(cells, grids=[grid])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="names",
            columns=[
                SimpleNamespace(
                    name="name", cell_id=None, bbox=None, description="名前"
                ),
            ],
        )

        result = _extract_table_field(info, field_schema)
        # header row should be skipped, only data row remains
        values = [r["name"]["value"] for r in result.value if "name" in r]
        assert "太郎" in values
        assert "名前" not in values

    def test_table_extraction_skips_header_row_multi_col(self):
        """複数列のグリッドでヘッダー行が正しくスキップされる"""
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        cells = {
            "h1": _make_cell("h1", [0, 0, 50, 30], "商品名", role="header"),
            "h2": _make_cell("h2", [50, 0, 100, 30], "金額", role="header"),
            "c1": _make_cell("c1", [0, 30, 50, 60], "りんご"),
            "c2": _make_cell("c2", [50, 30, 100, 60], "100"),
        }
        grid = self._make_grid(
            col_headers=[["h1"], ["h2"]],
            data=[["h1", "h2"], ["c1", "c2"]],  # first row is headers
        )
        table = _make_table(cells, grids=[grid])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="items",
            columns=[
                SimpleNamespace(
                    name="product", cell_id=None, bbox=None, description="商品名"
                ),
                SimpleNamespace(
                    name="price", cell_id=None, bbox=None, description="金額"
                ),
            ],
        )

        result = _extract_table_field(info, field_schema)
        assert len(result.value) == 1
        assert result.value[0]["product"]["value"] == "りんご"
        assert result.value[0]["price"]["value"] == "100"

    def test_no_columns_returns_not_found(self):
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        info = _make_semantic_info()
        field_schema = SimpleNamespace(name="items", columns=None)

        result = _extract_table_field(info, field_schema)
        assert result.source == "not_found"
        assert result.value == []

    def test_table_elements_have_labels(self):
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        cells = {
            "h1": _make_cell("h1", [0, 0, 50, 30], "商品", role="header"),
            "c1": _make_cell("c1", [0, 30, 50, 60], "りんご"),
        }
        grid = self._make_grid(
            col_headers=[["h1"]],
            data=[["c1"]],
        )
        table = _make_table(cells, grids=[grid])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="items",
            columns=[
                SimpleNamespace(
                    name="product", cell_id=None, bbox=None, description="商品"
                ),
            ],
        )

        result = _extract_table_field(info, field_schema)
        assert len(result.elements) == 1
        assert result.elements[0].label == "product"

    def test_duplicate_headers_concatenated(self):
        """同一ヘッダーが複数列に存在する場合、値を連結しcell_idsを複数格納する"""
        from yomitoku.extractor.rule_pipeline import _extract_table_field

        cells = {
            "h1": _make_cell("h1", [0, 0, 50, 30], "氏名", role="header"),
            "h2": _make_cell("h2", [50, 0, 100, 30], "生年月日", role="header"),
            "h3": _make_cell("h3", [100, 0, 150, 30], "生年月日", role="header"),
            "c1": _make_cell("c1", [0, 30, 50, 60], "太郎"),
            "c2": _make_cell("c2", [50, 30, 100, 60], "令和"),
            "c3": _make_cell("c3", [100, 30, 150, 60], "5年1月1日"),
        }
        grid = self._make_grid(
            col_headers=[["h1"], ["h2"], ["h3"]],
            data=[["c1", "c2", "c3"]],
        )
        table = _make_table(cells, grids=[grid])
        info = _make_semantic_info(tables=[table])

        field_schema = SimpleNamespace(
            name="people",
            columns=[
                SimpleNamespace(
                    name="name", cell_id=None, bbox=None, description="氏名"
                ),
                SimpleNamespace(
                    name="birth_date", cell_id=None, bbox=None, description="生年月日"
                ),
            ],
        )

        result = _extract_table_field(info, field_schema)
        assert result.source == "grid"
        assert len(result.value) == 1
        row = result.value[0]
        # 氏名は1列のみ
        assert row["name"]["value"] == "太郎"
        assert row["name"]["cell_ids"] == ["c1"]
        # 生年月日は2列が連結される
        assert row["birth_date"]["value"] == "令和5年1月1日"
        assert row["birth_date"]["cell_ids"] == ["c2", "c3"]
        # elementsにも両方のセルが含まれる
        birth_elements = [e for e in result.elements if e.label == "birth_date"]
        assert len(birth_elements) == 2
        assert {e.id for e in birth_elements} == {"c2", "c3"}


# =========================================================================
# pipeline.py - _normalize_resolved_fields
# =========================================================================
class TestNormalizeResolvedFields:
    def test_scalar_normalization(self):
        from yomitoku.extractor.pipeline import _normalize_resolved_fields
        from yomitoku.extractor.resolver import ResolvedField
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {"name": "phone", "type": "string", "normalize": "phone_jp"},
                ]
            }
        )

        fields = [
            ResolvedField(name="phone", value="09012345678", raw_text="09012345678")
        ]

        result = _normalize_resolved_fields(fields, schema)
        assert result[0].value == "090-1234-5678"

    def test_kv_structure_normalization(self):
        from yomitoku.extractor.pipeline import _normalize_resolved_fields
        from yomitoku.extractor.resolver import ResolvedField
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {
                        "name": "phone",
                        "type": "string",
                        "structure": "kv",
                        "normalize": "phone_jp",
                    },
                ]
            }
        )

        fields = [
            ResolvedField(name="phone", value="09012345678", raw_text="09012345678")
        ]

        result = _normalize_resolved_fields(fields, schema)
        assert result[0].value == "090-1234-5678"

    def test_skip_normalize(self):
        from yomitoku.extractor.pipeline import _normalize_resolved_fields
        from yomitoku.extractor.resolver import ResolvedField
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {"name": "phone", "type": "string", "normalize": "phone_jp"},
                ]
            }
        )

        fields = [
            ResolvedField(name="phone", value="09012345678", raw_text="09012345678")
        ]

        result = _normalize_resolved_fields(fields, schema, skip_normalize=True)
        assert result[0].value == "09012345678"

    def test_table_column_normalization(self):
        from yomitoku.extractor.pipeline import _normalize_resolved_fields
        from yomitoku.extractor.resolver import ResolvedField
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {
                        "name": "items",
                        "structure": "table",
                        "columns": [
                            {"name": "price", "type": "number", "normalize": "numeric"},
                        ],
                    }
                ]
            }
        )

        fields = [
            ResolvedField(
                name="items",
                value=[{"price": {"value": "１，２００円", "cell_ids": ["c1"]}}],
                raw_text="",
            )
        ]

        result = _normalize_resolved_fields(fields, schema)
        assert result[0].value[0]["price"]["value"] == "1200"


# =========================================================================
# pipeline.py - _build_output
# =========================================================================
class TestBuildOutput:
    def test_scalar_output(self):
        from yomitoku.extractor.pipeline import _build_output
        from yomitoku.extractor.resolver import ResolvedElement, ResolvedField

        fields = [
            ResolvedField(
                name="phone",
                value="03-1234-5678",
                raw_text="0312345678",
                elements=[
                    ResolvedElement(
                        id="c1", box=[10, 20, 30, 40], contents="0312345678"
                    )
                ],
                confidence="high",
                source="kv",
            )
        ]

        output = _build_output(fields)
        assert "fields" in output
        assert "metadata" in output
        phone_field = output["fields"]["phone"]
        assert phone_field["structure"] == "kv"
        assert phone_field["value"] == "03-1234-5678"
        assert phone_field["raw_text"] == "0312345678"
        assert phone_field["cell_ids"] == ["c1"]
        assert phone_field["bboxes"] == [[10, 20, 30, 40]]

    def test_table_output(self):
        from yomitoku.extractor.pipeline import _build_output
        from yomitoku.extractor.resolver import ResolvedElement, ResolvedField

        fields = [
            ResolvedField(
                name="items",
                value=[
                    {
                        "product": {"value": "apple", "cell_ids": ["c1"]},
                        "price": {"value": "100", "cell_ids": ["c2"]},
                    }
                ],
                raw_text="",
                elements=[
                    ResolvedElement(id="c1", box=[0, 0, 50, 30], contents="apple"),
                    ResolvedElement(id="c2", box=[50, 0, 100, 30], contents="100"),
                ],
                confidence="high",
                source="grid",
            )
        ]

        output = _build_output(fields)
        items = output["fields"]["items"]
        assert items["structure"] == "table"
        assert len(items["records"]) == 1
        row = items["records"][0]
        assert row["product"]["value"] == "apple"
        assert row["product"]["bboxes"] == [[0, 0, 50, 30]]
        assert row["price"]["value"] == "100"


# =========================================================================
# pipeline.py - _build_simple_output
# =========================================================================
class TestBuildSimpleOutput:
    def test_scalar_output(self):
        from yomitoku.extractor.pipeline import _build_simple_output
        from yomitoku.extractor.resolver import ResolvedElement, ResolvedField

        fields = [
            ResolvedField(
                name="phone",
                value="03-1234-5678",
                raw_text="0312345678",
                elements=[
                    ResolvedElement(
                        id="c1", box=[10, 20, 30, 40], contents="0312345678"
                    )
                ],
                confidence="high",
                source="kv",
            )
        ]

        output = _build_simple_output(fields)
        assert output == {"phone": "03-1234-5678"}

    def test_table_output(self):
        from yomitoku.extractor.pipeline import _build_simple_output
        from yomitoku.extractor.resolver import ResolvedElement, ResolvedField

        fields = [
            ResolvedField(
                name="items",
                value=[
                    {
                        "product": {"value": "apple", "cell_ids": ["c1"]},
                        "price": {"value": "100", "cell_ids": ["c2"]},
                    },
                    {
                        "product": {"value": "banana", "cell_ids": ["c3"]},
                        "price": {"value": "200", "cell_ids": ["c4"]},
                    },
                ],
                raw_text="",
                elements=[
                    ResolvedElement(id="c1", box=[0, 0, 50, 30], contents="apple"),
                    ResolvedElement(id="c2", box=[50, 0, 100, 30], contents="100"),
                ],
                confidence="high",
                source="grid",
            )
        ]

        output = _build_simple_output(fields)
        assert output == {
            "items": [
                {"product": "apple", "price": "100"},
                {"product": "banana", "price": "200"},
            ]
        }

    def test_mixed_scalar_and_table(self):
        from yomitoku.extractor.pipeline import _build_simple_output
        from yomitoku.extractor.resolver import ResolvedField

        fields = [
            ResolvedField(name="title", value="Invoice", raw_text="Invoice"),
            ResolvedField(
                name="items",
                value=[{"product": {"value": "pen", "cell_ids": ["c1"]}}],
                raw_text="",
            ),
        ]

        output = _build_simple_output(fields)
        assert output["title"] == "Invoice"
        assert output["items"] == [{"product": "pen"}]

    def test_scalar_dict_value_flattened(self):
        """LLMがKVフィールドをオブジェクトとして返した場合でもvalueを抽出する"""
        from yomitoku.extractor.pipeline import _build_simple_output
        from yomitoku.extractor.resolver import ResolvedField

        fields = [
            ResolvedField(
                name="model_code",
                value={
                    "name": "機種コード",
                    "value": "100031582200",
                    "cell_ids": ["c6"],
                    "confidence": "high",
                },
                raw_text="",
            )
        ]

        output = _build_simple_output(fields)
        assert output["model_code"] == "100031582200"


# =========================================================================
# prompt.py
# =========================================================================
class TestBuildMessages:
    def _make_minimal_semantic_info(self):
        """Create a real TableSemanticParserSchema-like mock for prompt building."""
        cell = _make_cell("c0", [0, 0, 100, 50], "test_value")
        para = _make_paragraph("p0", [0, 50, 100, 80], "paragraph text")
        word = _make_word("word_text", [[0, 0], [50, 0], [50, 20], [0, 20]])

        table = SimpleNamespace(
            id="t0",
            style="border",
            cells={"c0": cell},
            kv_items=[],
            grids=[],
        )

        info = SimpleNamespace(
            tables=[table],
            paragraphs=[para],
            words=[word],
        )
        return info

    def test_build_messages_returns_two_messages(self):
        from yomitoku.extractor.prompt import build_messages
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {"fields": [{"name": "test_field", "description": "テスト"}]}
        )

        info = self._make_minimal_semantic_info()
        messages = build_messages(info, schema)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_content(self):
        from yomitoku.extractor.prompt import build_messages
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {"fields": [{"name": "test_field", "description": "テスト"}]}
        )

        info = self._make_minimal_semantic_info()
        messages = build_messages(info, schema)

        system = messages[0]["content"]
        assert "document data extraction" in system
        assert "cell_ids" in system
        assert "structure=kv" in system
        assert "structure=table" in system

    def test_user_prompt_contains_sections(self):
        from yomitoku.extractor.prompt import build_messages
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {"fields": [{"name": "test_field", "description": "テスト"}]}
        )

        info = self._make_minimal_semantic_info()
        messages = build_messages(info, schema)

        user = messages[1]["content"]
        assert "## Paragraphs" in user
        assert "## Tables" in user
        assert "## Words" in user
        assert "## Extraction Schema" in user
        assert "## Response Format" in user

    def test_user_prompt_includes_data(self):
        from yomitoku.extractor.prompt import build_messages
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {"fields": [{"name": "test_field", "description": "テスト"}]}
        )

        info = self._make_minimal_semantic_info()
        messages = build_messages(info, schema)

        user = messages[1]["content"]
        assert "test_value" in user
        assert "paragraph text" in user
        assert "word_text" in user
        assert "test_field" in user

    def test_response_format_uses_actual_field_names(self):
        from yomitoku.extractor.prompt import build_messages
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {"name": "company_name", "description": "会社名"},
                    {"name": "phone", "description": "電話番号"},
                    {
                        "name": "items",
                        "structure": "table",
                        "columns": [
                            {"name": "product", "type": "string"},
                            {"name": "price", "type": "number"},
                        ],
                    },
                ]
            }
        )

        info = self._make_minimal_semantic_info()
        messages = build_messages(info, schema)
        user = messages[1]["content"]

        # Response format should contain actual field names, not placeholders
        assert '"company_name"' in user
        assert '"phone"' in user
        assert '"items"' in user
        assert '"product"' in user
        assert '"price"' in user
        # Should not contain abstract placeholders
        assert "<field_name>" not in user

    def test_table_id_filter(self):
        from yomitoku.extractor.prompt import _build_tables_section

        table0 = SimpleNamespace(
            id="t0",
            style="border",
            cells={"c0": _make_cell("c0", [0, 0, 10, 10], "t0_text")},
            kv_items=[],
            grids=[],
        )
        table1 = SimpleNamespace(
            id="t1",
            style="border",
            cells={"c1": _make_cell("c1", [0, 0, 10, 10], "t1_text")},
            kv_items=[],
            grids=[],
        )

        section = _build_tables_section([table0, table1], table_id_filter="t0")
        assert "t0_text" in section
        assert "t1_text" not in section

    def test_kv_items_contain_cell_ids(self):
        from yomitoku.extractor.prompt import _build_tables_section

        cells = {
            "c0": _make_cell("c0", [0, 0, 50, 30], "会社名", role="header"),
            "c1": _make_cell("c1", [50, 0, 150, 30], "ABC Corp"),
        }
        kv = SimpleNamespace(key=["c0"], value="c1")
        table = SimpleNamespace(
            id="t0", style="border", cells=cells, kv_items=[kv], grids=[]
        )

        section = _build_tables_section([table])
        # KV format: "- key_text: value_text [key_id,value_id]"
        assert "会社名: ABC Corp [c0,c1]" in section

    def test_grid_rows_contain_cell_ids(self):
        from yomitoku.extractor.prompt import _build_tables_section

        cells = {
            "h0": _make_cell("h0", [0, 0, 50, 30], "商品名", role="header"),
            "h1": _make_cell("h1", [50, 0, 100, 30], "数量", role="header"),
            "c0": _make_cell("c0", [0, 30, 50, 60], "Product A"),
            "c1": _make_cell("c1", [50, 30, 100, 60], "10"),
        }
        grid = SimpleNamespace(
            id="g0",
            col_headers=[["h0"], ["h1"]],
            data=[["c0", "c1"]],
        )
        table = SimpleNamespace(
            id="t0", style="border", cells=cells, kv_items=[], grids=[grid]
        )

        section = _build_tables_section([table])
        assert "Grid g0:" in section
        # Format: "header: value [value_ids]"
        assert "商品名: Product A [c0]" in section
        assert "数量: 10 [c1]" in section

    def test_grid_duplicate_headers_merged_in_prompt(self):
        """同一ヘッダーの複数列がプロンプト上で1エントリにマージされる"""
        from yomitoku.extractor.prompt import _build_tables_section

        cells = {
            "h0": _make_cell("h0", [0, 0, 50, 30], "氏名", role="header"),
            "h1": _make_cell("h1", [50, 0, 100, 30], "生年月日", role="header"),
            "h2": _make_cell("h2", [100, 0, 150, 30], "生年月日", role="header"),
            "c0": _make_cell("c0", [0, 30, 50, 60], "太郎"),
            "c1": _make_cell("c1", [50, 30, 100, 60], "令和"),
            "c2": _make_cell("c2", [100, 30, 150, 60], "5年1月1日"),
        }
        grid = SimpleNamespace(
            id="g0",
            col_headers=[["h0"], ["h1"], ["h2"]],
            data=[["c0", "c1", "c2"]],
        )
        table = SimpleNamespace(
            id="t0", style="border", cells=cells, kv_items=[], grids=[grid]
        )

        section = _build_tables_section([table])
        # 氏名 is unique
        assert "氏名: 太郎 [c0]" in section
        # 生年月日 columns are merged: values concatenated, IDs listed together
        assert "生年月日: 令和5年1月1日 [c1,c2]" in section
        # Should NOT have separate 生年月日 entries
        assert section.count("生年月日") == 1

    def test_only_unassigned_cells_listed(self):
        from yomitoku.extractor.prompt import _build_tables_section

        cells = {
            "c0": _make_cell("c0", [0, 0, 50, 30], "会社名", role="header"),
            "c1": _make_cell("c1", [50, 0, 150, 30], "ABC Corp"),
            "c2": _make_cell("c2", [0, 30, 150, 60], "orphan text"),
            "c3": _make_cell("c3", [0, 60, 150, 90], "group text", role="group"),
        }
        kv = SimpleNamespace(key=["c0"], value="c1")
        table = SimpleNamespace(
            id="t0", style="border", cells=cells, kv_items=[kv], grids=[]
        )

        section = _build_tables_section([table])
        # c0 and c1 are referenced by KV, should NOT be in Unassigned
        assert "Unassigned Cells:" in section
        assert "orphan text" in section
        # c3 is a group cell, should be excluded entirely
        assert "group text" not in section
        # c0/c1 appear in KV section, not in Unassigned
        lines = section.split("\n")
        unassigned_lines = []
        in_unassigned = False
        for line in lines:
            if "Unassigned Cells:" in line:
                in_unassigned = True
                continue
            if in_unassigned:
                unassigned_lines.append(line)
        assert not any("c0:" in line for line in unassigned_lines)
        assert not any("c1:" in line for line in unassigned_lines)
        assert any("c2:" in line for line in unassigned_lines)


# =========================================================================
# prompt.py - _build_schema_section
# =========================================================================
class TestBuildSchemaSection:
    def test_scalar_field(self):
        from yomitoku.extractor.prompt import _build_schema_section
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {"fields": [{"name": "phone", "description": "電話番号", "type": "string"}]}
        )

        section = _build_schema_section(schema)
        assert "phone" in section
        assert "電話番号" in section
        assert "structure=kv" in section
        assert "type=string" in section

    def test_kv_structure_field(self):
        from yomitoku.extractor.prompt import _build_schema_section
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {
                        "name": "phone",
                        "description": "電話番号",
                        "structure": "kv",
                        "type": "string",
                    }
                ]
            }
        )

        section = _build_schema_section(schema)
        assert "structure=kv" in section
        assert "structure=table" not in section

    def test_table_field(self):
        from yomitoku.extractor.prompt import _build_schema_section
        from yomitoku.extractor.schema import ExtractionSchema

        schema = ExtractionSchema.model_validate(
            {
                "fields": [
                    {
                        "name": "items",
                        "structure": "table",
                        "columns": [
                            {"name": "product", "type": "string"},
                            {"name": "price", "type": "number"},
                        ],
                    }
                ]
            }
        )

        section = _build_schema_section(schema)
        assert "items" in section
        assert "structure=table" in section
        assert "product" in section
        assert "price" in section


# =========================================================================
# normalizer.py - _parse_date (edge cases)
# =========================================================================
class TestParseDate:
    def test_meiji(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("明治45年7月30日") == "1912-07-30"

    def test_taisho(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("大正15年12月25日") == "1926-12-25"

    def test_abbrev_T(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("T15/12/25") == "1926-12-25"

    def test_abbrev_M(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("M45.7.30") == "1912-07-30"

    def test_fullwidth_digits(self):
        from yomitoku.extractor.normalizer import date_jp

        assert date_jp("令和６年３月１５日") == "2024-03-15"

    def test_time_jp_fullwidth(self):
        from yomitoku.extractor.normalizer import time_jp

        assert time_jp("１４：３０") == "14時30分"

    def test_time_hms_fullwidth(self):
        from yomitoku.extractor.normalizer import time_hms

        assert time_hms("１４時３０分") == "14:30:00"
