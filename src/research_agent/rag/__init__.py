"""RAG bonus: hybrid evidence retrieval (BM25 + embeddings) used by the Critic."""
from .embeddings import EmbeddingChain, get_embedder, set_embedder
from .hybrid import Doc, RetrievalResult, chunk_text, retrieve

__all__ = ["Doc", "EmbeddingChain", "RetrievalResult", "chunk_text", "get_embedder", "retrieve", "set_embedder"]
