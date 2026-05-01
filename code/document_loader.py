"""
document_loader.py
------------------
Loads Sanskrit documents from .txt, .pdf, and .docx files.
Returns a list of raw document strings ready for preprocessing.
"""

import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_txt(filepath: str) -> str:
    """Read a plain-text file (UTF-8, handles Devanagari)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_docx(filepath: str) -> str:
    """
    Extract text from a .docx file using python-docx.
    Paragraphs are joined with newlines.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_pdf(filepath: str) -> str:
    """
    Extract text from a PDF using pdfplumber.
    Falls back to PyPDF2 if pdfplumber is unavailable.
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        pass

    try:
        import PyPDF2
        text_parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install pdfplumber\n"
            "  pip install PyPDF2"
        )


_LOADERS = {
    ".txt":  load_txt,
    ".docx": load_docx,
    ".pdf":  load_pdf,
}


def load_document(filepath: str) -> Dict[str, str]:
    """
    Load a single document and return a dict with 'source' and 'text' keys.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the document file.

    Returns
    -------
    dict with keys:
        source (str): filename
        text   (str): full extracted text
    """
    ext = os.path.splitext(filepath)[-1].lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_LOADERS.keys())}"
        )

    logger.info(f"Loading document: {filepath}")
    text = _LOADERS[ext](filepath)
    logger.info(f"  → {len(text)} characters extracted.")

    return {"source": os.path.basename(filepath), "text": text}


def load_directory(directory: str) -> List[Dict[str, str]]:
    """
    Recursively load all supported documents from a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing Sanskrit documents.

    Returns
    -------
    List of dicts, each with 'source' and 'text'.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[-1].lower()
            if ext in _LOADERS:
                fpath = os.path.join(root, fname)
                try:
                    doc = load_document(fpath)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")

    logger.info(f"Loaded {len(documents)} document(s) from '{directory}'.")
    return documents
