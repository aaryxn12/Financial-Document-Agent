"""
Extracts text (and tables where possible) from an uploaded PDF.
Strategy: try pdfplumber's table extraction first (better for financial
statements); fall back to pypdf plain-text extraction if no tables are
found or the PDF has none. Returns an error/empty-signal cleanly rather
than raising, so the app can show a friendly message instead of crashing.
"""
