"""
FastAPI layer exposing the Financial Document Agent's core RAG pipeline as
a REST API.

State handling: each uploaded document's chunk texts are persisted to S3
(api/document_store.py), since Lambda gives no guarantee that two requests
for the same document_id land on the same execution environment. The
in-memory Chroma collection itself is never persisted -- Chroma re-derives
identical embeddings from the same chunk text deterministically, so it's
cheaper to rebuild than to serialize. _documents acts as a warm-cache: fast
when a later request happens to land on the same warm environment that
handled the upload or a prior question, falling back to an S3 fetch plus a
fresh collection rebuild otherwise.
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
from api.document_store import save_chunks, load_chunks

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
    save_chunks(document_id, chunks)

    return UploadResponse(document_id=document_id, chunk_count=len(chunks))


@app.post("/documents/{document_id}/ask", response_model=AskResponse)
async def ask_document(document_id: str, request: AskRequest) -> AskResponse:
    collection = _documents.get(document_id)
    if collection is None:
        chunks = load_chunks(document_id)
        if chunks is None:
            raise HTTPException(status_code=404, detail="Unknown document_id.")
        collection = create_collection()
        add_chunks(collection, chunks)
        _documents[document_id] = collection

    result = run_agent(collection, request.question)
    return AskResponse(answer=result.answer, sources=result.sources, iterations=result.iterations)

handler = Mangum(app)