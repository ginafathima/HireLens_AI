"""Document parsing utilities for extracting text from PDF and DOCX resumes.

Provides modular parsers with comprehensive error handling for corrupted files,
unsupported formats, and empty documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import docx
from pypdf import PdfReader
from pypdf.errors import PyPdfError


class ParsingError(Exception):
    """Base exception for resume parsing errors."""


class UnsupportedFormatError(ParsingError):
    """Raised when the submitted file extension is not supported."""


class EmptyDocumentError(ParsingError):
    """Raised when the document contains no readable text."""


class CorruptDocumentError(ParsingError):
    """Raised when the document cannot be read or is corrupted."""


# pylint: disable=too-few-public-methods


class BaseResumeParser(ABC):
    """Abstract base class for resume document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Extract text from the specified document.

        Args:
            file_path: Path to the target document.

        Returns:
            str: Raw extracted text.

        Raises:
            ParsingError: If extraction fails.
        """


class PDFParser(BaseResumeParser):
    """Parser for PDF resumes using pypdf."""

    def parse(self, file_path: Path) -> str:
        """Extract text page-by-page from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Extracted text joined by newlines.

        Raises:
            FileNotFoundError: If the file does not exist.
            CorruptDocumentError: If the PDF is malformed or unreadable.
            EmptyDocumentError: If no text could be extracted.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            reader = PdfReader(str(file_path))
            if reader.is_encrypted:
                try:
                    # Attempt empty password decryption for unprotected encrypted PDFs
                    reader.decrypt("")
                except PyPdfError as err:
                    raise CorruptDocumentError(
                        f"PDF is encrypted with a password and cannot be read: {file_path.name}"
                    ) from err

            extracted_chunks = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_chunks.append(page_text.strip())

            full_text = "\n\n".join(extracted_chunks).strip()
            if not full_text:
                raise EmptyDocumentError(
                    f"PDF document contains no readable text: {file_path.name}. "
                    "It may be a scanned image or completely blank."
                )
            return full_text

        except (EmptyDocumentError, CorruptDocumentError):
            raise
        except Exception as exc:
            raise CorruptDocumentError(
                f"Failed to read PDF file '{file_path.name}': {str(exc)}"
            ) from exc


class DocxParser(BaseResumeParser):
    """Parser for DOCX resumes using python-docx."""

    def parse(self, file_path: Path) -> str:
        """Extract text from paragraphs and tables in a DOCX file.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            str: Extracted text joined by newlines.

        Raises:
            FileNotFoundError: If the file does not exist.
            CorruptDocumentError: If the DOCX is malformed or unreadable.
            EmptyDocumentError: If no text could be extracted.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        try:
            doc = docx.Document(str(file_path))
            extracted_chunks = []

            # 1. Extract from standard paragraphs
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    extracted_chunks.append(text)

            # 2. Extract from tables (resumes often format skills/education in tables)
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    # Deduplicate repeated merged cell text in the same row
                    unique_texts = []
                    for t in row_texts:
                        if not unique_texts or t != unique_texts[-1]:
                            unique_texts.append(t)
                    if unique_texts:
                        extracted_chunks.append(" | ".join(unique_texts))

            full_text = "\n".join(extracted_chunks).strip()
            if not full_text:
                raise EmptyDocumentError(
                    f"DOCX document contains no readable text: {file_path.name}."
                )
            return full_text

        except (EmptyDocumentError, CorruptDocumentError):
            raise
        except Exception as exc:
            raise CorruptDocumentError(
                f"Failed to read DOCX file '{file_path.name}': {str(exc)}"
            ) from exc


def parse_resume(file_path: Union[str, Path]) -> str:
    """Convenience dispatcher to parse a resume by its file extension.

    Args:
        file_path: Path to the resume file (.pdf or .docx).

    Returns:
        str: Cleanly extracted text from the document.

    Raises:
        FileNotFoundError: If the target file does not exist.
        UnsupportedFormatError: If the file extension is not .pdf or .docx.
        ParsingError: If extraction fails or document is empty.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parser = PDFParser()
    elif suffix == ".docx":
        parser = DocxParser()
    else:
        raise UnsupportedFormatError(
            f"Unsupported file format '{suffix}'. Only .pdf and .docx files are supported."
        )

    return parser.parse(path)
