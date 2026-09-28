"""
Streamlit UI for the Financial Document Agent: upload a financial PDF and
ask questions about it, with sourced answers.

Streamlit's execution model is different from a typical request/response
app (like the FastAPI layer) or a component-based frontend -- the ENTIRE
script re-runs top to bottom on every user interaction (every button click,
every chat message sent). Nothing survives between reruns except what's
explicitly stashed in `st.session_state`. Conceptually it's close to
Angular re-running a component's render logic on every state change --
except here the "component" is the whole script, every time.

This matters concretely: without the session_state guard below, uploading
a document once and then sending five chat messages would re-extract,
re-chunk, and re-embed the ENTIRE PDF five times, once per rerun -- a real,
easy-to-miss performance bug that a lot of first-time Streamlit code has.

Each browser tab gets its own `st.session_state`, which is what makes an
in-memory Chroma collection safe here for a multi-user hosted demo -- one
visitor's uploaded document never leaks into another visitor's session.
"""

import re

import streamlit as st

from core.extraction import extract_text
from core.chunking import chunk_text
from core.retrieval import create_collection, add_chunks
from core.agent import run_agent


def _sanitize_for_display(text: str) -> str:
    """Clean up the model's raw answer text before rendering it with
    st.markdown() -- two independent issues found by comparing rendered
    output against result.answer's raw repr():

    1. Inline citation markers ([chunk_2] / \u3010chunk_2\u3011) -- the Sources
       expander already shows these cleanly, so they're stripped from the
       displayed prose.
    2. Streamlit's markdown renderer supports LaTeX math via $...$ (a
       well-known Streamlit gotcha). A currency figure like "$45,781" has
       an unescaped $, and an answer with several dollar amounts lets
       pairs of $ signs get greedily matched as math-mode delimiters,
       corrupting the bold/italic markdown around and between them.
       Confirmed live: the raw model text was fully well-formed (every **
       pair matched, no missing whitespace) even though the RENDERED
       output showed broken, glued-together words -- proving the
       corruption happens in Streamlit's renderer, not the model's output.
       Escaping every literal $ as \\$ stops it being read as a math
       delimiter without changing how the number looks.
    """
    cleaned = re.sub(r"[\[\u3010][^\]\u3011]*chunk_\d+[^\]\u3011]*[\]\u3011]", "", text)
    cleaned = re.sub(r"\s+([.,])", r"\1", cleaned)  # fix " ." left behind
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.replace("$", "\\$")
    return cleaned.strip()


st.set_page_config(
    page_title="Financial Document Agent",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 760px; }
    .stChatMessage { padding: 0.75rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Financial Document Agent")
st.caption(
    "Upload a financial statement (10-Q, 10-K, earnings release) and ask "
    "questions about it. Answers are grounded in the document, with sources "
    "shown for every response."
)

# --- session state setup -------------------------------------------------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role", "content", "sources"?}]

# --- sidebar: upload -------------------------------------------------------
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.document_name:
        with st.spinner("Reading and indexing document..."):
            text = extract_text(uploaded_file)
            if not text:
                st.error("Couldn't extract any text from this PDF.")
                st.session_state.collection = None
                st.session_state.document_name = None
            else:
                chunks = chunk_text(text)
                collection = create_collection()
                add_chunks(collection, chunks)

                st.session_state.collection = collection
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []  # fresh chat for a new document

    if st.session_state.document_name:
        st.success(f"Indexed **{st.session_state.document_name}** ({st.session_state.chunk_count} chunks)")
    else:
        st.info("Upload a PDF to get started.")

# --- main: chat -------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content_to_show = message["content"]
        if message["role"] == "assistant":
            content_to_show = _sanitize_for_display(content_to_show)
        st.markdown(content_to_show)
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                st.write(", ".join(message["sources"]))

question = st.chat_input(
    "Ask a question about the document..."
    if st.session_state.collection is not None
    else "Upload a document first"
)

if question:
    if st.session_state.collection is None:
        st.warning("Upload a document before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                result = run_agent(st.session_state.collection, question)
            st.markdown(_sanitize_for_display(result.answer))
            if result.sources:
                with st.expander(f"Sources ({len(result.sources)})"):
                    st.write(", ".join(result.sources))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )