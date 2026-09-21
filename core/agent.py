"""
The agentic RAG loop: a search_documents tool wrapping retrieval.py,
a tool schema, and a ReAct-style loop (topic 7) that lets the model
decide when/how many times to search before answering with citations.
Generalized to reference "the uploaded document" rather than a
specific hardcoded company/filing.
"""
