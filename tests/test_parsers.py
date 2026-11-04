"""Unit tests for document parsers (PDF and DOCX)."""

from pathlib import Path
import pytest

from src.parsers import (
    parse_resume,
    UnsupportedFormatError,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_resumes"


def test_parse_strong_pdf_resume():
    """Verify that candidate_strong.pdf extracts cleanly with key terms intact."""
    pdf_path = SAMPLE_DIR / "candidate_strong.pdf"
    assert pdf_path.exists(), "candidate_strong.pdf must exist"

    text = parse_resume(pdf_path)
    assert len(text) > 200
    assert "AARAV SHARMA" in text.upper()
    assert "Python" in text
    assert "pypdf" in text
    assert "FastAPI" in text


def test_parse_weak_docx_resume():
    """Verify that candidate_weak.docx extracts cleanly with table content intact."""
    docx_path = SAMPLE_DIR / "candidate_weak.docx"
    assert docx_path.exists(), "candidate_weak.docx must exist"

    text = parse_resume(docx_path)
    assert len(text) > 200
    assert "ROHAN PATEL" in text.upper()
    assert "Excel" in text
    assert "Accounting" in text


def test_parser_file_not_found():
    """Verify FileNotFoundError is raised when target does not exist."""
    with pytest.raises(FileNotFoundError):
        parse_resume("sample_resumes/non_existent_file.pdf")


def test_unsupported_format():
    """Verify UnsupportedFormatError is raised for non-PDF/DOCX extensions."""
    test_file = SAMPLE_DIR / "temp_file.txt"
    test_file.write_text("plain text", encoding="utf-8")
    try:
        with pytest.raises(UnsupportedFormatError):
            parse_resume(test_file)
    finally:
        test_file.unlink(missing_ok=True)
