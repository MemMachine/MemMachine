
from typing import Tuple
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

def extract_text_from_pdf(path: str) -> str:
    try:
        return pdf_extract_text(path)
    except Exception:
        return ""

def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text_from_txt(path: str, encoding: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def sniff_and_read(path: str, filename: str) -> Tuple[str, str]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(path), "pdf"
    if name.endswith(".docx"):
        return extract_text_from_docx(path), "docx"
    if name.endswith(".txt") or name.endswith(".md"):
        return extract_text_from_txt(path), "text"
    # Fallback try text
    return extract_text_from_txt(path), "unknown"
