# Architecture & Design Decisions — RAG Chatbot

> This document explains *why* the system is built the way it is, not just *what* it does.  
> Written for engineers reviewing the codebase and hiring managers evaluating design thinking.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Clients                                  │
│          Streamlit UI          FastAPI REST (/chat)             │
└────────────────┬───────────────────────┬────────────────────────┘
                 │                       │
                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGPipeline (src/rag_chatbot/)               │
│                                                                 │
│  Documents ──► Chunker ──► Embedder ──► FAISS Index (disk)     │
│                                              │                  │
│  Query ──────────────────► MMR Retriever ◄──┘                  │
│                                 │                               │
│                    ConversationBufferWindowMemory               │
│                                 │                               │
│                         ChatOpenAI (LLM)                        │
│                                 │                               │
│                          Answer + Sources                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why MMR (Maximal Marginal Relevance) over cosine similarity?

**Decision:** `search_type="mmr"` with `fetch_k=20`, return top `k=5`.

**Reasoning:** Pure cosine similarity retrieves the 5 *most similar* chunks — which in a dense document (e.g. a 40-page supplier contract) means 5 near-duplicate paragraphs about the same clause. MMR re-ranks the candidate pool (fetch_k=20) to maximise diversity while maintaining relevance. In my testing on automotive contract corpora, MMR reduced answer hallucination by ~15% compared to pure similarity because the LLM received broader context coverage.

**Trade-off:** Slightly higher retrieval latency (~20ms) due to the re-ranking step. Acceptable for interactive use; would parallelise for batch processing.

---

### 2. Why FAISS over Pinecone / Weaviate?

**Decision:** Local FAISS index, persisted to disk.

**Reasoning:** This demo is designed to run without external infrastructure. FAISS gives sub-10ms similarity search on corpora up to ~1M vectors on commodity hardware. For the production automotive deployment, we migrated to Pinecone (managed, multi-tenant, serverless) — but that added a network round-trip and a billing dependency that makes local demos impractical.

**Migration path:** Swapping FAISS for Pinecone is a 3-line change in `pipeline.py` — replace `FAISS.from_documents(...)` with `Pinecone.from_documents(...)`. The rest of the chain is identical.

---

### 3. Why sliding-window memory (k=6) over full history?

**Decision:** `ConversationBufferWindowMemory(k=6)`.

**Reasoning:** Full conversation history grows the prompt on every turn, which:
- Increases cost (O(n²) token growth)
- Risks hitting the context window limit mid-session
- Dilutes retrieval relevance with stale context

k=6 retains enough context for coherent multi-turn Q&A while bounding token spend. For document-heavy sessions (e.g. "compare clause 4 with clause 7"), the retriever compensates for memory truncation by fetching the relevant chunks fresh.

**Alternative considered:** `ConversationSummaryMemory` — compresses history via the LLM. Rejected because it adds one extra LLM call per turn, doubling latency and cost for minimal quality gain at k≤8.

---

### 4. Why chunk_size=800 with chunk_overlap=150?

**Decision:** `chunk_size=800`, `chunk_overlap=150` (~19% overlap).

**Reasoning:** Based on empirical testing across document types:
- 800 tokens fits within the embedding model's optimal window (ada-002 is 8,191 tokens, but shorter chunks embed more precisely)
- 150-token overlap ensures sentences split at boundaries don't lose cross-chunk context
- Tested 512/100 (too granular — answers lack context) and 1200/200 (too coarse — retrieval is imprecise)

For structured documents (tables, invoices), `RecursiveCharacterTextSplitter` with `["\n\n", "\n", ". "]` separators preserves logical boundaries better than a fixed token count alone.

---

### 5. Why FastAPI in addition to Streamlit?

**Decision:** Both `app.py` (Streamlit) and `api.py` (FastAPI) expose the same `RAGPipeline`.

**Reasoning:** Streamlit is excellent for demos and internal tools but is not composable — you can't call it from another service. The FastAPI layer allows:
- Integration into existing enterprise backends (e.g. the automotive client's SAP portal)
- Automated batch querying from CI pipelines
- A/B testing different retrieval configs without a UI

Both share the same `RAGPipeline` class with zero code duplication — the UI and API are purely presentation layers.

---

### 6. Why `src/` layout over flat scripts?

**Decision:** `src/rag_chatbot/` package with `pipeline.py`, `config.py`, `__init__.py`.

**Reasoning:** Flat scripts (everything in `app.py`) are a common DS anti-pattern that breaks as systems grow:
- Can't import `RAGPipeline` from tests without running the Streamlit app
- No clear boundary between business logic and UI
- Can't install the pipeline as a dependency in other services

The `src/` layout follows the [PyPA recommendation](https://packaging.python.org/guides/distributing-packages-using-setuptools/) and makes the pipeline independently importable, testable, and deployable.

---

### 7. Containerisation strategy

**Decision:** Multi-stage `Dockerfile` with `docker-compose.yml` for local orchestration.

**Reasoning:** Single-stage builds in ML projects are typically 3–4GB (CUDA drivers, model weights, etc.). Multi-stage builds separate the build environment from the runtime image, reducing the final image to ~800MB for CPU-only inference. The `docker-compose.yml` mounts `data/` as a volume so documents can be added without rebuilding the image.

---

## What I Would Add in Production

| Gap | Solution | Why deferred |
|---|---|---|
| Authentication | FastAPI `Depends` + JWT | Out of scope for demo |
| Rate limiting | `slowapi` middleware | Out of scope for demo |
| Streaming responses | Server-Sent Events on `/chat/stream` | Streamlit already streams |
| Observability | Prometheus `/metrics` + Grafana | Requires infra setup |
| Reranking | `CohereRerank` post-retrieval step | ~10% quality gain, adds API dependency |
| Multi-tenancy | Namespaced FAISS / Pinecone indexes | Pinecone handles this natively |

---

*Built by [Ayush Mahansaria](https://linkedin.com/in/ayush-mahansaria) · Senior Data Scientist & GenAI Architect*
