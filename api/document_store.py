"""
S3-backed persistence for uploaded documents' chunk text.

The API runs as a Lambda function, which gives no guarantee that two
requests for the same document_id land on the same execution environment
(or that either request lands on a warm one at all) -- see
core/retrieval.py's docstring for why the Chroma collection itself is
never what gets persisted. Chroma re-derives identical embeddings from
the same chunk text deterministically, so the only thing that actually
needs to survive between invocations is the plain list of chunk strings;
a fresh collection is cheaply rebuilt from that at query time.

The Streamlit deployment never imports this module -- it keeps its
collection in st.session_state for the life of a user's browser session,
which is sufficient there since Streamlit runs as one long-lived process,
not a per-invocation Lambda function.
"""

import json

import boto3

_BUCKET = "financial-document-agent-085787041728-ap-south-1-an"
_s3 = boto3.client("s3", region_name="ap-south-1")


def save_chunks(document_id: str, chunks: list[str]) -> None:
    """Persist a document's chunk texts to S3, keyed by document_id."""
    _s3.put_object(
        Bucket=_BUCKET,
        Key=f"documents/{document_id}.json",
        Body=json.dumps(chunks).encode("utf-8"),
        ContentType="application/json",
    )


def load_chunks(document_id: str) -> list[str] | None:
    """Fetch a document's chunk texts from S3, or None if not found."""
    try:
        response = _s3.get_object(Bucket=_BUCKET, Key=f"documents/{document_id}.json")
    except _s3.exceptions.NoSuchKey:
        return None
    return json.loads(response["Body"].read())