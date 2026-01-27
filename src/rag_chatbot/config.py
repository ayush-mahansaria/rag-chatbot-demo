"""
Centralised configuration for the RAG pipeline.
All tunable parameters live here — no magic numbers scattered through the code.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = 5
    fetch_k: int = 20          # MMR candidate pool size
    search_type: str = "mmr"   # "mmr" | "similarity" | "similarity_score_threshold"

    # ── Models ────────────────────────────────────────────────────
    llm_model: str = "gpt-3.5-turbo"
    embed_model: str = "text-embedding-ada-002"
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_temperature: float = 0.2
    use_local_embeddings: bool = False

    # ── Memory ────────────────────────────────────────────────────
    memory_window_k: int = 6   # sliding-window turns to retain

    # ── Storage ───────────────────────────────────────────────────
    vector_db_path: str = "vectorstore/faiss_index"
    data_dir: str = "data/"

    # ── Separators for RecursiveCharacterTextSplitter ─────────────
    separators: list = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    def vector_db_exists(self) -> bool:
        return Path(f"{self.vector_db_path}.faiss").exists()
