"""rag_chatbot — production RAG pipeline package."""
from .pipeline import RAGPipeline
from .config import RAGConfig

__all__ = ["RAGPipeline", "RAGConfig"]
__version__ = "1.1.0"
