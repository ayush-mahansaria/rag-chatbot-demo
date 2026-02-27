# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] — 2024-03-15

### Added
- `src/rag_chatbot/` package — proper Python module structure replacing flat `app.py`
- `RAGConfig` dataclass — all tunable parameters centralised, no magic numbers
- `docs/ARCHITECTURE.md` — detailed design decision documentation
- `docker-compose.yml` — single-command local deployment (`docker compose up`)
- Multi-stage `Dockerfile` — ~800MB runtime image vs ~3GB naive build
- FastAPI `/stats` endpoint — session-level token usage and latency analytics
- FastAPI `/memory` DELETE endpoint — reset conversation history without restart

### Changed
- Retrieval upgraded from cosine similarity to **MMR** (Maximal Marginal Relevance)
- Memory upgraded from full-history buffer to **sliding-window** (k=6)
- Tests reorganised into `tests/test_pipeline.py` and `tests/test_api.py`

### Fixed
- FAISS index now persists across container restarts via mounted volume

---

## [1.1.0] — 2024-02-28

### Added
- FastAPI REST layer (`api.py`) — `/chat`, `/ingest`, `/health` endpoints
- GitHub Actions CI — runs pytest on Python 3.10 + 3.11 on every push
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
