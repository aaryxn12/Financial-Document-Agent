# Financial Document Agent

Upload a financial PDF (10-K, earnings release, quarterly statement) and ask questions about it in plain English — get back grounded answers with citations to the exact source text, powered by a retrieval-augmented, tool-calling LLM agent.

Built as the Phase 2 capstone project of a self-directed LLM/RAG/agentic-systems curriculum.

## Architecture

- `core/` — the actual pipeline logic (extraction, chunking, retrieval, agent loop), framework-agnostic
- `api/` — FastAPI layer exposing `core/` as a REST API (run locally with `uvicorn api.main:app --reload`; see `/docs` for the interactive API spec)
- `streamlit_app.py` — the hosted demo UI, imports `core/` directly (no network hop) for simple free-tier deployment

## Setup

1. `uv venv` / `python -m venv .venv` and activate it
2. `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)
3. Add your Groq API key to `.env`: `GROQ_API_KEY=...`
4. Run the UI: `streamlit run streamlit_app.py`
5. (Optional) Run the API: `uvicorn api.main:app --reload`

## Status

Work in progress — Phase 2 capstone, in active development.

## Known limitations / future improvements

- **Chunking** uses fixed-size character chunking with overlap, not boundary-aware splitting — it can occasionally slice a word or number across a chunk boundary. A production upgrade path would be sentence/paragraph-aware chunking (e.g. LangChain's `RecursiveCharacterTextSplitter`). Kept hand-rolled deliberately for this project to demonstrate the underlying mechanics rather than relying on a framework.
- **Extraction** combines `pypdf` and `pdfplumber` output for redundancy (header/context text vs. structured table rows), which means the raw pre-chunking text contains some duplication — acceptable for correctness, not optimized for token efficiency.
