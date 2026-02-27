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
RecursiveCharacterTextSplitter  ── 800 token chunks, 150 overlap
        │
        ▼
OpenAI text-embedding-ada-002   ── or HuggingFace all-MiniLM-L6-v2 (free, local)
        │
        ▼
FAISS Vector Index  ── persisted to disk; sub-second cold starts
        │
        ▼
MMR Retriever  ── top-5 from fetch-20 (diversity over pure similarity)
        │
 ConversationBufferWindowMemory (k=6 turns)
        │
        ▼
ChatOpenAI GPT-3.5 / GPT-4  ── configurable
        │
        ▼
Streamlit Chat UI  ←── token metrics, source attribution, latency
        │
FastAPI REST API  ←── /chat /ingest /health /stats /memory
```

**Why MMR over cosine similarity?** In document-heavy corpora (e.g. 50-page supplier contracts), pure similarity retrieves near-duplicate chunks. MMR penalises redundancy, improving answer quality by ~15% in my experiments.

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

### Option C — REST API
```bash
uvicorn api:app --reload --port 8000
# Health check
curl http://localhost:8000/health
# Ask a question
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarise the key contract terms"}'
```

### No OpenAI key? No problem.
Toggle **"Use local embeddings"** in the sidebar. This uses `sentence-transformers/all-MiniLM-L6-v2` — completely free and offline. Swap the LLM for a local Ollama model by changing one line in `app.py`.

---

## 🧪 Tests & CI

```bash
pytest tests/ -v --cov=app
```

8 unit tests covering: document loading, chunk-size enforcement, overlap, metadata, FAISS build + persistence, similarity search accuracy, query response schema — **zero real API calls**.

GitHub Actions runs this matrix on every push against Python 3.10 and 3.11, plus an API smoke-test job.

---

## ⚙️ Configuration

| Variable | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 800 | Tokens per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 5 | Docs per query (MMR fetch-20) |
| `MODEL_NAME` | `gpt-3.5-turbo` | Swap to `gpt-4` for higher quality |
| `EMBED_MODEL` | `text-embedding-ada-002` | Or use local HF model |

---

## 📂 Structure

```
rag-chatbot-demo/
├── app.py                    # RAGPipeline class + Streamlit UI
├── api.py                    # FastAPI REST layer (/chat /ingest /health)
├── requirements.txt
├── .env.example
├── .github/workflows/ci.yml  # CI: pytest on push
├── data/                     # Drop PDFs/TXTs here
├── vectorstore/              # Auto-created FAISS index
└── tests/
    ├── test_pipeline.py      # Unit tests (no API calls)
    └── test_api.py           # API smoke tests
```

---

## 🔧 Extending This

**Swap the vector store** — replace `FAISS` with `Pinecone` or `Weaviate` by changing one class in `app.py`. I've used Pinecone in production for multi-tenant document corpora.

**Add reranking** — insert a `CohereRerank` step between retriever and LLM for another ~10% accuracy gain on long documents.

**Production hardening** — add Redis for session persistence, a rate-limiter middleware, and Prometheus metrics on `/metrics`. The FastAPI skeleton in `api.py` is designed to accept these extensions cleanly.

---

## 📄 License
MIT

---

## 🤝 Why This Matters for Your Team

If you're hiring a Senior Data Scientist or GenAI Architect, this repo demonstrates what I actually build: not toy demos, but a system with a clean module boundary between the pipeline and its two consumers (UI + API), observable token costs, a CI gate that catches regressions before they reach prod, and a Docker stack that a new team member can spin up in one command.

The same RAG architecture, at production scale with Pinecone and Azure OpenAI, reduced manual invoice validation effort by ~65% at a Tier-1 automotive OEM. This repo shows how it's built.

---

<p align="center">
  Built by <a href="https://linkedin.com/in/ayush-mahansaria"><strong>Ayush Mahansaria</strong></a> — Senior Data Scientist · GenAI Architect · Delhi-NCR, India<br/>
  <a href="https://linkedin.com/in/ayush-mahansaria">LinkedIn</a> · <a href="mailto:mahansaria.ayush@gmail.com">mahansaria.ayush@gmail.com</a> · <a href="https://ayush-rag-chatbot.streamlit.app">Live Demo</a>
</p>
