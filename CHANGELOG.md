# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.3.0] — 2024-03-22

### Changed
- **LLM upgraded** from `gpt-3.5-turbo` → `gpt-4o-mini`
  - 128k context window (vs 16k) — handles larger retrieved chunks without truncation
  - Lower cost per token than gpt-3.5-turbo
  - Significantly better structured document Q&A quality
- **Embedding model upgraded** from `text-embedding-ada-002` → `text-embedding-3-large`
  - Higher MTEB benchmark scores (~54.9 vs ~48.4)
  - Better retrieval precision on dense technical documents
- **Chunk size increased** from 800 → 1000 tokens (leverages gpt-4o-mini's larger context)
- **TOP_K_RETRIEVAL increased** from 5 → 8 (more context, better completeness)

### Fixed
- LangChain imports updated to correct package locations for langchain>=0.2
  - `langchain_classic.chains` → `langchain.chains`
  - `langchain_classic.memory` → `langchain.memory`
  - `OpenAIEmbeddings`, `ChatOpenAI` → `langchain_community`

---

## [1.2.0] — 2024-03-15

### Added
- `src/rag_chatbot/` package — proper Python module structure replacing flat `app.py`
- `RAGConfig` dataclass — all tunable parameters centralised
- `docs/ARCHITECTURE.md` — 7 design decisions with explicit trade-off analysis
- `docker-compose.yml` — single-command local deployment (`docker compose up`)
- Multi-stage `Dockerfile` — ~800MB runtime image vs ~3GB naive build
- FastAPI `/stats` endpoint — session-level token usage and latency analytics
- FastAPI `/memory` DELETE endpoint — reset conversation without restart
- Issue templates for bugs and feature requests

### Changed
- Retrieval upgraded from cosine similarity to **MMR** (Maximal Marginal Relevance)
- Memory upgraded from full-history buffer to **sliding-window** (k=6)

### Fixed
- FAISS index now persists across container restarts via mounted volume

---

## [1.1.0] — 2024-02-28

### Added
- FastAPI REST layer (`api.py`) — `/chat`, `/ingest`, `/health` endpoints
- GitHub Actions CI — pytest on Python 3.10 + 3.11 on every push
- Local embedding fallback (HuggingFace `all-MiniLM-L6-v2`) — runs without API key
- `.env.example` — clear setup instructions for new contributors

### Changed
- README rewritten with architecture diagram, live demo badge, design rationale
- `requirements.txt` pinned to tested versions

---

## [1.0.0] — 2024-02-01

### Added
- Initial release — `RAGPipeline` class with ingestion, chunking, FAISS, OpenAI
- Streamlit chat UI with source attribution and token tracking
- Unit tests for core pipeline (no API calls)
