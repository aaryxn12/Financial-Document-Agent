"""
Extracts text (and tables where possible) from an uploaded PDF.

Strategy: run pypdf's plain-text extraction and pdfplumber's table-aware
extraction and combine both -- pypdf reliably keeps narrative context
(titles, column headers) that pdfplumber's table detection can swallow;
pdfplumber keeps numeric table rows grouped together in a way plain
text extraction loses. Returns "" only if neither method extracts
anything, so the caller can show a friendly "couldn't read this PDF"
message instead of crashing or proceeding with an empty document.
"""

import pdfplumber
from pypdf import PdfReader


def extract_text_pdfplumber(file) -> str:
    """
    Table-aware extraction. Financial statements are usually tabular,
    so tables are pulled out first
    """
    parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    parts.append(" | ".join(cleaned_row))
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
    return "\n".join(parts).strip()


def extract_text_pypdf(file) -> str:
    """Fallback: plain-text extraction, no table structure awareness."""
    reader = PdfReader(file)
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts).strip()


def extract_text(file) -> str:
    """
    Extract text from an uploaded PDF file-like object, combining both
    methods rather than picking one -- pypdf reliably keeps narrative
    context (titles, column headers like "Three Months Ended...") that
    pdfplumber's table detection can swallow, while pdfplumber keeps
    numeric rows grouped together in a way plain text extraction loses.
    Returns "" only if BOTH methods fail to extract anything.
    """

    #tested this pypdf & pdfplumber on Apple's Financial data and pypdf gave better results
    file.seek(0)
    try:
        plain_text = extract_text_pypdf(file)
    except Exception:
        plain_text = ""

    file.seek(0)
    try:
        table_text = extract_text_pdfplumber(file)
    except Exception:
        table_text = ""

    return "\n\n".join(part for part in (plain_text, table_text) if part)
