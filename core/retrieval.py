"""
Embeds chunks and stores/queries them in a Chroma collection.

For the hosted app, each user/session gets its own fresh, in-memory
Chroma client (chromadb.Client(), not PersistentClient) -- no data
persists across sessions and nothing leaks between concurrent users,
since free hosting doesn't offer real persistent disk anyway (see
README). Embedding happens automatically via Chroma's default
embedding function (sentence-transformers' all-MiniLM-L6-v2, same
model used in topic 5 practice).
"""

import uuid
import chromadb


def create_collection():
    """
    Creates a fresh, in-memory Chroma client plus a new collection
    inside it, scoped to one uploaded document. Call this once per
    upload (e.g. store the returned collection in st.session_state so
    it survives Streamlit reruns for that user, but never gets shared
    with anyone else's session).
    """
    client = chromadb.Client()
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
    return client.create_collection(name=collection_name)


def add_chunks(collection, chunks: list[str]) -> None:
    """
    Embeds and stores a list of text chunks in the given collection.
    Chunk IDs are simple positional labels (chunk_0, chunk_1, ...) so
    the agent can cite them back to the user later.
    """
    if not chunks:
        raise ValueError("No chunks to add -- extraction/chunking may have failed upstream")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)


def query_collection(collection, query: str, n_results: int = 3) -> tuple[list[str], list[str]]:
    """
    Queries the collection for the n_results chunks most relevant to
    query. Returns (chunk_ids, chunk_texts) in matching order, so the
    agent can label each chunk with its ID when building citations.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["ids"][0], results["documents"][0]
