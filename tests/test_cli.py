import os
from pathlib import Path

import pytest

from yomitoku.cli import main
from yomitoku.utils.logger import set_logger
from yomitoku.cli.main import validate_encoding

logger = set_logger(__name__, "DEBUG")


def test_run_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "tests/data/dummy",
            "-o",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileNotFoundError):
        main.main()


def test_run_not_exist_cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "tests/data/test.jpg",
            "-o",
            str(tmp_path),
            "--tr_cfg",
            "tests/yaml/dummy.yaml",
        ],
    )
    with pytest.raises(FileNotFoundError):
        main.main()


def test_run_txt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "tests/data/test.txt",
            "-o",
            str(tmp_path),
        ],
    )
    with pytest.raises(ValueError):
        main.main()


def test_run_invalid_format(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "tests/data/test.pdf",
            "-o",
            str(tmp_path),
            "-f",
            "invalid",
        ],
    )
    with pytest.raises(ValueError):
        main.main()


def test_run_png_markdown(monkeypatch, tmp_path):
    path_img = "tests/data/test.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            path_img,
            "-o",
            str(tmp_path),
            "-f",
            "markdown",
            "-v",
            "--td_cfg",
            "tests/yaml/text_detector.yaml",
        ],
    )
    main.main()
    path = Path(path_img)
    dirname = path.parent.name
    filename = path.stem
    out_path = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1.md")
    out_ocr = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1_ocr.jpg")
    out_lay = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1_layout.jpg")
    assert os.path.exists(out_path)
    assert os.path.exists(out_ocr)
    assert os.path.exists(out_lay)


def test_run_jpg_html(monkeypatch, tmp_path):
    path_img = "tests/data/test.jpg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            path_img,
            "-o",
            str(tmp_path),
            "-f",
            "HTML",
            "--lp_cfg",
            "tests/yaml/layout_parser.yaml",
            "--figure",
            "--figure_letter",
            "--figure_width",
            "100",
            "--ignore_line_break",
            "--lite",
        ],
    )
    main.main()
    path = Path(path_img)
    dirname = path.parent.name
    filename = path.stem
    out_path = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1.html")
    assert os.path.exists(out_path)


def test_run_tiff_csv(monkeypatch, tmp_path):
    path_img = "tests/data/test.tiff"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            path_img,
            "-o",
            str(tmp_path),
            "-f",
            "csv",
            "--tsr_cfg",
            "tests/yaml/table_structure_recognizer.yaml",
            "--lite",
        ],
    )
    main.main()
    path = Path(path_img)
    dirname = path.parent.name
    filename = path.stem
    out_path = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1.csv")
    assert os.path.exists(out_path)


def test_run_pdf_md(monkeypatch, tmp_path):
    path_img = "tests/data/test.pdf"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            path_img,
            "-o",
            str(tmp_path),
            "-f",
            "MD",
            "--tr_cfg",
            "tests/yaml/text_recognizer.yaml",
            "--figure",
            "--figure_letter",
            "--figure_width",
            "100",
            "--ignore_line_break",
            "--lite",
        ],
    )
    main.main()
    path = Path(path_img)
    dirname = path.parent.name
    filename = path.stem
    out_path_p1 = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1.md")
    out_path_p2 = os.path.join(str(tmp_path), f"{dirname}_{filename}_p2.md")
    assert os.path.exists(out_path_p1)
    assert os.path.exists(out_path_p2)


def test_run_dir_json(monkeypatch, tmp_path):
    path_img = "tests/data"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            path_img,
            "-o",
            str(tmp_path),
            "-f",
            "json",
        ],
    )
    main.main()
    path = Path(path_img)
    dirname = path.name
    filename = "test"
    out_path = os.path.join(str(tmp_path), f"{dirname}_{filename}_p1.json")
    assert os.path.exists(out_path)


def test_validate_encoding():
    with pytest.raises(ValueError):
        validate_encoding("utf-9")

    assert validate_encoding("utf-8")
    assert validate_encoding("shift-jis")
    assert validate_encoding("euc-jp")
    assert validate_encoding("cp932")
