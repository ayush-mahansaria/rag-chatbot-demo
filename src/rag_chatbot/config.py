"""
Centralised configuration for the RAG pipeline.
All tunable parameters live here — no magic numbers scattered through the code.

Current production config (as of v1.2):
  - Model:      gpt-4o-mini    (128k context, cost-efficient)
  - Embeddings: text-embedding-3-large  (higher MTEB accuracy than ada-002)
  - Chunk size: 1000 tokens    (optimised for gpt-4o-mini context window)
  - Top-K:      8 chunks       (leverages extended context vs old k=4)

See docs/ARCHITECTURE.md for full rationale on each decision.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = 1000       # Larger than ada-era default; gpt-4o-mini handles it
    chunk_overlap: int = 150     # ~15% overlap prevents boundary info loss

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = 8               # More chunks = better coverage with 128k context
    fetch_k: int = 20            # MMR candidate pool size
    search_type: str = "mmr"     # "mmr" | "similarity" | "similarity_score_threshold"

    # ── Models ────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"               # 128k ctx, cheaper + better than gpt-3.5-turbo
    embed_model: str = "text-embedding-3-large"  # Higher MTEB accuracy than ada-002
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_temperature: float = 0.2
    use_local_embeddings: bool = False

    # ── Memory ────────────────────────────────────────────────────
    memory_window_k: int = 6     # Sliding window — bounds O(n²) token growth

    # ── Storage ───────────────────────────────────────────────────
    vector_db_path: str = "vectorstore/faiss_index"
    data_dir: str = "data/"

    # ── Separators for RecursiveCharacterTextSplitter ─────────────
    separators: list = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    def vector_db_exists(self) -> bool:
        return Path(f"{self.vector_db_path}.faiss").exists()
