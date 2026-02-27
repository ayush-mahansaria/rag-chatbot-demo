"""rag_chatbot — production RAG pipeline package."""
from .pipeline import RAGPipeline
from .config import RAGConfig
from .query_rewriter import rewrite_query

__all__ = ["RAGPipeline", "RAGConfig", "rewrite_query"]
__version__ = "1.4.0"
