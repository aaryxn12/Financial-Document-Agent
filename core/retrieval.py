"""
Embeds chunks and stores/queries them in a Chroma collection.
For the hosted app, each user session gets its own fresh, in-memory
Chroma client (chromadb.Client(), not PersistentClient) — no data
persists across sessions or leaks between concurrent users.
"""
