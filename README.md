# 🤖 RAG Chatbot Demo

[![CI](https://github.com/ayush-mahansaria/rag-chatbot-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/ayush-mahansaria/rag-chatbot-demo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ayush-rag-chatbot.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **[▶ Live Demo →](https://ayush-rag-chatbot.streamlit.app)**

---

This is a production-grade **Retrieval-Augmented Generation pipeline** — the same architecture I deployed for a Tier-1 Polish automotive manufacturer to automate supplier invoice validation and contract compliance checking. In that engagement, processing an invoice went from a 4-step manual workflow to a single API call, reducing analyst validation effort by ~65%.

This repo exposes the full technical stack as an interactive demo you can run against your own documents in under 5 minutes.

---

## 📐 Architecture

```
Documents (PDF / TXT)
        │
        ▼
RecursiveCharacterTextSplitter  ── 1000 token chunks, 150 overlap
        │
        ▼
text-embedding-3-large (OpenAI)  ── or HuggingFace all-MiniLM-L6-v2 (free, local)
        │
        ▼
FAISS Vector Index  ── persisted to disk; sub-second cold starts
        │
        ▼
MMR Retriever  ── top-8 from fetch-20 (diversity over pure similarity)
        │
 ConversationBufferWindowMemory (k=6 turns)
        │
        ▼
gpt-4o-mini  ── 128k context window, cost-efficient, fast
        │
        ▼
Streamlit Chat UI  ←── token metrics, source attribution, latency
        │
FastAPI REST API  ←── /chat /ingest /health /stats /memory
```

**Why `text-embedding-3-large` over `ada-002`?**
`text-embedding-3-large` scores significantly higher on MTEB benchmarks. For dense document corpora (e.g. 50-page supplier contracts), better embeddings mean the right chunks surface on the first retrieval pass — reducing the need for post-retrieval reranking and cutting hallucination.

**Why `gpt-4o-mini` over `gpt-3.5-turbo`?**
`gpt-4o-mini` has a 128k context window (vs 16k), handles longer retrieved chunks without truncation, is cheaper per token than `gpt-3.5-turbo`, and produces significantly better answers on structured document Q&A tasks.

**Why MMR over cosine similarity?**
Pure cosine similarity retrieves the most similar chunks — often near-duplicate paragraphs. MMR re-ranks a 20-candidate pool to maximise diversity while preserving relevance. In my testing on automotive contract corpora, MMR reduced hallucination by ~15% by giving the LLM broader context coverage.

**Why chunk_size=1000 with overlap=150?**
1000 tokens sits comfortably within the embedding model's optimal window while providing enough context per chunk for multi-sentence reasoning. 150-token overlap prevents information loss at chunk boundaries. Tested against 512 (too granular) and 1500 (retrieval too coarse).

---

## 🚀 Quick Start

### Option A — Streamlit Cloud (no setup)
Click **[▶ Live Demo](https://ayush-rag-chatbot.streamlit.app)** and upload any PDF or TXT.

### Option B — Run locally
```bash
git clone https://github.com/ayush-mahansaria/rag-chatbot-demo.git
cd rag-chatbot-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
mkdir -p data/ && cp your_docs.pdf data/
streamlit run app.py
```

### Option C — Docker (one command)
```bash
docker compose up
# UI → http://localhost:8501
# API → http://localhost:8000/docs
```

### Option D — REST API only
```bash
uvicorn api:app --reload --port 8000
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarise the key contract terms"}'
```

### No OpenAI key?
Toggle **"Use local embeddings"** in the sidebar — uses `sentence-transformers/all-MiniLM-L6-v2`, completely free and offline.

---

## 📊 Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `MODEL_NAME` | `gpt-4o-mini` | 128k context, cost-efficient, better than gpt-3.5-turbo |
| `EMBED_MODEL` | `text-embedding-3-large` | Higher MTEB accuracy than ada-002 |
| `CHUNK_SIZE` | `1000` tokens | Optimal for gpt-4o-mini's extended context |
| `CHUNK_OVERLAP` | `150` tokens | Prevents information loss at boundaries |
| `TOP_K_RETRIEVAL` | `8` | More chunks retrieved for higher accuracy |
| `MMR fetch_k` | `20` | Candidate pool for diversity reranking |
| `Memory window` | `k=6` turns | Bounds token cost while preserving context |

---

## 🧪 Tests & CI

```bash
pytest tests/ -v
```

GitHub Actions runs the full test matrix on Python 3.10 and 3.11 on every push. Tests cover the API layer and config validation without making real OpenAI calls.

---

## 📂 Project Structure

```
rag-chatbot-demo/
├── src/rag_chatbot/          # Importable pipeline package
│   ├── __init__.py
│   ├── config.py             # All tunable parameters in one place
│   └── pipeline.py           # Core RAGPipeline class
├── tests/
│   ├── conftest.py           # Path setup for CI
│   ├── test_pipeline.py
│   └── test_api.py
├── docs/
│   └── ARCHITECTURE.md       # 7 design decisions with trade-off analysis
├── .github/
│   ├── workflows/ci.yml      # pytest on Python 3.10 + 3.11
│   └── ISSUE_TEMPLATE/
├── app.py                    # Streamlit chat UI
├── api.py                    # FastAPI REST layer
├── Dockerfile                # Multi-stage, ~800MB runtime image
├── docker-compose.yml        # API + UI with one command
├── pyproject.toml            # Package config + linting rules
├── requirements.txt
└── README.md
```

---

## 🔧 Extending This

**Swap the vector store** — replace `FAISS` with `Pinecone` or `Weaviate` by changing one class. I've used Pinecone in production for multi-tenant document corpora.

**Add reranking** — insert `CohereRerank` between retriever and LLM for another ~10% accuracy gain on long documents.

**Upgrade the LLM** — swap `gpt-4o-mini` for `gpt-4o` in `MODEL_NAME` for higher quality on complex reasoning tasks. The rest of the chain is unchanged.

**Production hardening** — add Redis for session persistence, a rate-limiter middleware, and Prometheus metrics on `/metrics`. The FastAPI skeleton in `api.py` is designed to accept these extensions cleanly.

---

## 🤝 Why This Matters for Your Team

If you're hiring a Senior Data Scientist or GenAI Architect, this repo demonstrates what I actually build: a clean module boundary between pipeline and consumers (UI + API), observable token costs, CI that catches regressions before they reach prod, and a Docker stack a new team member can spin up in one command.

The same RAG architecture — at production scale with Pinecone, Azure OpenAI, and `gpt-4o` — reduced manual invoice validation effort by ~65% at a Tier-1 automotive OEM. This repo shows exactly how it's built.

---

<p align="center">
  Built by <a href="https://linkedin.com/in/ayush-mahansaria"><strong>Ayush Mahansaria</strong></a> — Senior Data Scientist · GenAI Architect · Delhi-NCR, India<br/>
  <a href="https://linkedin.com/in/ayush-mahansaria">LinkedIn</a> · <a href="mailto:mahansaria.ayush@gmail.com">mahansaria.ayush@gmail.com</a> · <a href="https://ayush-rag-chatbot.streamlit.app">Live Demo</a>
</p>

## 📄 License
MIT
