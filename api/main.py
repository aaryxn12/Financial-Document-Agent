"""
FastAPI layer exposing the Financial Document Agent's core RAG pipeline as
a REST API.

State handling: each uploaded document gets its own in-memory Chroma
collection, held in a process-local dict keyed by a generated document ID.
This is intentionally simple -- fine for a single-instance demo deployment,
but uploaded documents don't survive a server restart and this won't work
correctly behind multiple worker processes (each worker would have its own
empty dict). A production version would move this to a shared store (e.g.
a persistent Chroma instance keyed by document ID, or Redis for the
mapping) instead of process memory.
"""

import io
import uuid
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from mangum import Mangum

from core.extraction import extract_text
from core.chunking import chunk_text
from core.retrieval import create_collection, add_chunks
from core.agent import run_agent

app = FastAPI(title="Financial Document Agent API")

# document_id -> Chroma collection. Process-local; see module docstring.
_documents: Dict[str, object] = {}


class UploadResponse(BaseModel):
    document_id: str
    chunk_count: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    iterations: int


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    text = extract_text(io.BytesIO(contents))
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract any text from this PDF.")

    chunks = chunk_text(text)
    collection = create_collection()
    add_chunks(collection, chunks)

    document_id = uuid.uuid4().hex
    _documents[document_id] = collection

    return UploadResponse(document_id=document_id, chunk_count=len(chunks))


@app.post("/documents/{document_id}/ask", response_model=AskResponse)
async def ask_document(document_id: str, request: AskRequest) -> AskResponse:
    collection = _documents.get(document_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Unknown document_id.")

    result = run_agent(collection, request.question)
    return AskResponse(answer=result.answer, sources=result.sources, iterations=result.iterations)

handler = Mangum(app)