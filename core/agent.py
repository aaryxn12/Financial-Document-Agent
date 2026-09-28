"""
Wraps retrieval.py's query_collection as a tool inside a ReAct loop, so the
LLM decides when and how often to search the document rather than always
retrieving a fixed top-k for every question (agentic RAG, not naive RAG).

The agent is intentionally generic -- the system prompt and tool description
refer to "the uploaded document," not any specific company or filing, since
this module has to work against whatever PDF a user uploads through the
API/Streamlit layers.

Sources returned to the caller are tracked server-side from the actual
retrieval calls, not parsed out of the model's citation text -- the model's
[chunk_N] citations in its answer are for human readability, but they aren't
a reliable signal to depend on programmatically (a model can format a
citation inconsistently, e.g. "【chunk_2】" instead of "[chunk_2]").
"""

from dataclasses import dataclass, field
import json
import os

from dotenv import load_dotenv
from groq import Groq

from core.retrieval import query_collection

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a financial document assistant. Answer questions \
about a document the user has uploaded, using ONLY information retrieved via \
the search_document tool -- never your own general knowledge or assumptions.

Rules:
- Call search_document to find relevant passages before answering. Call it \
again with a different query if the first search doesn't give you enough \
information.
- Every factual claim must be grounded in a retrieved passage. Cite the \
chunk ID(s) you used, e.g. [chunk_3].
- You may perform simple arithmetic (percentages, differences, sums) on \
numbers you retrieved, but never invent a number that wasn't retrieved.
- If the retrieved passages don't contain the answer after a reasonable \
number of searches, say so explicitly instead of guessing.
- Write your answer as plain prose in complete sentences. Do not use \
markdown tables, headers, or bullet lists -- if you need to present several \
figures, do it as a short sentence list (e.g. "Americas was $45,781M, \
Europe was $29,395M, ..."). You may use **bold** for key figures, but \
every bold or italic marker you open must be closed within the same \
sentence -- unmatched markdown formatting renders broken.
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search the uploaded document for passages relevant to a query. "
            "Returns the most relevant chunks along with their chunk IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query describing what to look for.",
                }
            },
            "required": ["query"],
        },
    },
}


@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    iterations: int = 0


def _search(collection, query: str, seen_sources: list[str]) -> str:
    ids, chunks = query_collection(collection, query)
    if not chunks:
        return "No relevant passages found."
    for cid in ids:
        if cid not in seen_sources:
            seen_sources.append(cid)
    return "\n\n".join(f"[{cid}] {chunk}" for cid, chunk in zip(ids, chunks))


def run_agent(collection, question: str) -> AgentResponse:
    """
    Runs the ReAct loop against `collection` for a single user `question`.
    `collection` is whatever core.retrieval.create_collection() returned for
    this session/document -- the caller (API or Streamlit layer) owns it.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    sources: list[str] = []

    for i in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            return AgentResponse(answer=msg.content, sources=sources, iterations=i)

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = _search(collection, args["query"], sources)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return AgentResponse(
        answer="I wasn't able to find a confident answer within the allowed number of searches.",
        sources=sources,
        iterations=MAX_ITERATIONS,
    )