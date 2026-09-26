"""
Document Parser Module
Extracts clean plain text from PDF, DOCX, and TXT files.
Supports both file paths and Streamlit UploadedFile (file-like) objects.
"""

import io
import os
from typing import Union
import PyPDF2

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def extract_text_from_pdf(source: Union[str, io.BytesIO, bytes]) -> str:
    """Extract text from a PDF file path or file-like stream."""
    text_content = []
    try:
        if isinstance(source, (str, os.PathLike)):
            with open(source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content.append(extracted)
        elif isinstance(source, bytes):
            stream = io.BytesIO(source)
            reader = PyPDF2.PdfReader(stream)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted)
        else:
            # Assume file-like object (BytesIO or Streamlit UploadedFile)
            reader = PyPDF2.PdfReader(source)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted)
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

    return "\n".join(text_content).strip()


def extract_text_from_docx(source: Union[str, io.BytesIO, bytes]) -> str:
    """Extract text from a DOCX file path or file-like stream."""
    if not HAS_DOCX:
        raise ImportError("python-docx is not installed. Please install it via pip install python-docx")

    text_content = []
    try:
        if isinstance(source, (str, os.PathLike)):
            doc = docx.Document(source)
        elif isinstance(source, bytes):
            stream = io.BytesIO(source)
            doc = docx.Document(stream)
        else:
            doc = docx.Document(source)

        for para in doc.paragraphs:
            if para.text.strip():
                text_content.append(para.text.strip())

        # Also extract text inside tables if any
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_content.append(" | ".join(row_text))

    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")

    return "\n".join(text_content).strip()


def extract_text_from_txt(source: Union[str, io.BytesIO, bytes]) -> str:
    """Extract text from a TXT file path or file-like stream with encoding detection."""
    try:
        if isinstance(source, (str, os.PathLike)):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        elif isinstance(source, bytes):
            return source.decode("utf-8", errors="replace").strip()
        else:
            # File-like object
            content = source.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace").strip()
            return str(content).strip()
    except Exception as e:
        raise ValueError(f"Error reading TXT file: {str(e)}")


def parse_document(file_source: Union[str, io.BytesIO, bytes], filename: str = None) -> str:
    """
    Unified entry point to parse a document based on file extension.
    If filename is not given and file_source is a string path, filename is inferred.
    If file_source is a Streamlit UploadedFile, filename is taken from file_source.name.
    """
    if hasattr(file_source, "name") and not filename:
        filename = file_source.name

    if isinstance(file_source, (str, os.PathLike)) and not filename:
        filename = str(file_source)

    if not filename:
        raise ValueError("Cannot determine file type without a filename.")

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_source)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_source)
    elif ext in [".txt", ".md", ".csv", ".py", ".java", ".c", ".cpp"]:
        return extract_text_from_txt(file_source)
    else:
        # Fallback to plain text attempt
        try:
            return extract_text_from_txt(file_source)
        except Exception:
            raise ValueError(f"Unsupported file format: {ext}")
