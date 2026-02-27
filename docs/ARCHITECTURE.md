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
│                    RAGPipeline (app.py / src/rag_chatbot/)      │
│                                                                 │
│  Documents ──► Chunker ──► text-embedding-3-large ──► FAISS    │
│                                      │                          │
│  Query ──────────────► MMR Retriever (k=8, fetch_k=20) ◄──────┘│
│                                 │                               │
│                  ConversationBufferWindowMemory (k=6)           │
│                                 │                               │
│                      gpt-4o-mini (128k ctx)                     │
│                                 │                               │
│                          Answer + Sources                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why `gpt-4o-mini` over `gpt-3.5-turbo`?

**Decision:** `MODEL_NAME = "gpt-4o-mini"`

**Reasoning:**

| Model | Context window | Cost (input) | Quality on doc Q&A |
|---|---|---|---|
| gpt-3.5-turbo | 16k tokens | $0.0015/1k | Baseline |
| gpt-4o-mini | **128k tokens** | **$0.00015/1k** | Significantly better |

`gpt-4o-mini` is both cheaper and better than `gpt-3.5-turbo`. The 128k context window is the key operational advantage — with `TOP_K_RETRIEVAL=8` and `CHUNK_SIZE=1000`, we can pass ~8,000 tokens of retrieved context to the LLM without truncation. On `gpt-3.5-turbo`, larger retrieval sets would exceed the context limit.

**Trade-off:** For the absolute highest quality on complex multi-document reasoning, `gpt-4o` outperforms `gpt-4o-mini`. `MODEL_NAME` is a single string swap when that quality level is needed.

---

### 2. Why `text-embedding-3-large` over `ada-002`?

**Decision:** `EMBED_MODEL = "text-embedding-3-large"`

**Reasoning:** On the MTEB (Massive Text Embedding Benchmark), `text-embedding-3-large` scores ~54.9 vs `ada-002`'s ~48.4 — a meaningful gap in retrieval precision. In practice on long documents (e.g. 40-page supplier contracts), better embeddings mean the correct clause surfaces on the first retrieval pass rather than requiring a second-pass reranker.

**Cost comparison:**
- `ada-002`: $0.0001 / 1k tokens
- `text-embedding-3-large`: $0.00013 / 1k tokens

~30% more expensive per embed call, but embeddings are computed once at ingest time and cached to the FAISS index. For a typical 50-page document (~25k tokens), this is a $0.003 one-time cost difference — negligible against the quality gain.

**Trade-off:** `text-embedding-3-large` has a slightly longer inference time per batch. For real-time single-query embedding (at query time), this is imperceptible. For bulk re-indexing of large corpora (>1M chunks), parallelise with `asyncio`.

---

### 3. Why `CHUNK_SIZE=1000` with `CHUNK_OVERLAP=150`?

**Decision:** 1000 token chunks, 150 token overlap (~15%).

**Reasoning:** Calibrated specifically for `gpt-4o-mini`'s 128k context window:

- **1000 tokens** fits within the embedding model's optimal window while providing enough context per chunk for multi-sentence reasoning. At `TOP_K=8`, we pass ~8,000 tokens to the LLM — well within the 128k limit.
- **150 token overlap** (~1–2 paragraphs) prevents information loss at chunk boundaries. A key sentence split across two chunks will appear in at least one chunk in full.
- Tested alternatives: `512/100` (too granular — answers lack paragraph-level context), `1500/200` (retrieval precision drops — chunks become too topically mixed).

`RecursiveCharacterTextSplitter` with `["\n\n", "\n", ". ", " ", ""]` separators respects logical document boundaries (section breaks, paragraphs, sentences) before falling back to word splits.

---

### 4. Why `TOP_K_RETRIEVAL=8` with `MMR fetch_k=20`?

**Decision:** Retrieve 8 final chunks via MMR from a 20-candidate pool.

**Reasoning:**
- **k=8** (vs the common k=4) leverages `gpt-4o-mini`'s extended context window. More context reduces the chance of missing relevant information when the answer spans multiple document sections.
- **MMR with fetch_k=20** first retrieves 20 candidates by cosine similarity, then reranks them to maximise diversity. This prevents 8 near-duplicate chunks (e.g. 8 versions of the same contract clause) from crowding out other relevant sections.
- In testing on multi-section contracts, `k=8` + MMR improved answer completeness by ~20% vs `k=4` + cosine similarity.

**Trade-off:** More retrieved chunks = higher token cost per query (~800–1200 additional tokens). Bounded by `gpt-4o-mini`'s low per-token cost.

---

### 5. Why sliding-window memory (`k=6`) over full history?

**Decision:** `ConversationBufferWindowMemory(k=6)`

**Reasoning:** Full conversation history grows the prompt O(n²) as the session continues. At turn 20, full history would add ~15,000+ tokens to every query. The `k=6` window retains enough context for coherent multi-turn Q&A (e.g. "compare that with what you said about clause 7") while bounding token cost.

When memory is truncated, the retriever compensates by fetching the relevant document chunks fresh — the user's earlier question context may drop out of memory, but the *document content* it referenced is always re-retrievable.

**Alternative considered:** `ConversationSummaryMemory` — compresses history via an LLM call. Rejected because it adds one extra LLM call per turn, roughly doubling latency and cost for minimal quality gain at `k<=8`.

---

### 6. Why FAISS over Pinecone / Weaviate?

**Decision:** Local FAISS index, persisted to disk.

**Reasoning:** This demo is designed to run without external infrastructure. FAISS delivers sub-10ms similarity search on corpora up to ~1M vectors on commodity hardware. It persists to disk as two files (`faiss_index.faiss` + `faiss_index.pkl`), enabling cold-start reloads in under a second.

**Production migration path:** Swapping to Pinecone is a 3-line change — replace `FAISS.from_documents(...)` with `Pinecone.from_documents(...)`. The rest of the chain is identical. In the automotive production deployment, we use Pinecone (managed, multi-tenant, serverless) because it handles concurrent users and index updates without downtime.

---

### 7. Why FastAPI in addition to Streamlit?

**Decision:** Both `app.py` (Streamlit) and `api.py` (FastAPI) expose the same `RAGPipeline`.

**Reasoning:** Streamlit is excellent for demos and internal tools but is not composable — you cannot call it from another service. The FastAPI layer enables:
- Integration into enterprise backends (e.g. SAP portal, internal ticketing system)
- Automated batch querying from CI/testing pipelines
- A/B testing different retrieval configs programmatically

Both share the same pipeline class with zero code duplication.

---

## Production Architecture (as deployed at automotive OEM)

```
Azure Data Factory
    └─ Document ingestion (daily delta from SharePoint + email attachments)
             │
             ▼
    Azure Blob Storage (raw documents)
             │
             ▼
    Azure Container Apps
    ├─ Ingestion worker (LangChain + text-embedding-3-large → Pinecone)
    └─ FastAPI query server (gpt-4o, k=8 MMR, sliding memory)
             │
             ▼
    SAP Portal integration (REST API → invoice validation workflow)
             │
             ▼
    Power BI dashboard (validation metrics, SLA tracking)
```

---

## What I Would Add Next

| Enhancement | Estimated Impact | Effort |
|---|---|---|
| `CohereRerank` post-retrieval | ~10% answer quality gain | 1 day |
| Streaming `/chat/stream` SSE | Better perceived latency | 1 day |
| Prometheus `/metrics` endpoint | Production observability | 1 day |
| Redis session persistence | Multi-user session isolation | 2 days |
| Upgrade to `gpt-4o` | Higher quality on complex docs | 5-minute config change |

---

*Built by [Ayush Mahansaria](https://linkedin.com/in/ayush-mahansaria) · Senior Data Scientist · GenAI Architect · Delhi-NCR, India*
