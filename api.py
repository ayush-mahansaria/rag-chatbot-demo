"""
FastAPI serving layer for the RAG Chatbot.
Exposes a REST API so the pipeline can be integrated into any backend,
CI system, or production orchestrator — not just the Streamlit UI.

Run locally:
    uvicorn api:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"question": "What is RAG?"}'
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy-import so the API starts fast even without full deps
try:
    from app import RAGPipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = logging.getLogger("uvicorn.error")

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, example="What is RAG?")
    stream: bool = Field(False, description="Reserved for future streaming support")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens_used: int
    latency_ms: int
    session_tokens: int

class IngestRequest(BaseModel):
    source_dir: str = Field("data/", description="Directory containing PDF/TXT files to ingest")

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    total_session_tokens: int
    rag_available: bool

class StatsResponse(BaseModel):
    total_queries: int
    total_tokens: int
    avg_latency_ms: float
    avg_tokens_per_query: float

# ─────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────
class AppState:
    pipeline: Optional["RAGPipeline"] = None
    total_queries: int = 0
    total_latency_ms: int = 0

app_state = AppState()

# ─────────────────────────────────────────────
# Lifespan — initialise pipeline on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise RAG pipeline. Shutdown: clean up."""
    logger.info("🚀 Starting RAG API — initialising pipeline…")
    if RAG_AVAILABLE:
        try:
            Path("data/").mkdir(exist_ok=True)
            # Seed demo content if data/ is empty
            demo = Path("data/demo.txt")
            if not any(Path("data/").iterdir()):
                demo.write_text(
                    "RAG (Retrieval-Augmented Generation) combines a retrieval system "
                    "with a generative language model. Documents are embedded into a "
                    "vector store; at query time, the most relevant chunks are retrieved "
                    "and passed to the LLM as context, grounding the response in real data."
                )
            pipeline = RAGPipeline(use_local_embeddings=False)
            pipeline.initialize("data/")
            app_state.pipeline = pipeline
            logger.info("✅ Pipeline ready.")
        except Exception as e:
            logger.warning(f"⚠️  Pipeline init failed (API will run in degraded mode): {e}")
    else:
        logger.warning("⚠️  RAG dependencies not installed — running in stub mode.")
    yield
    logger.info("👋 Shutting down RAG API.")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "Production-grade REST API for the Retrieval-Augmented Generation chatbot. "
        "Built with LangChain · FAISS · OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """Liveness + readiness check."""
    return HealthResponse(
        status="ok",
        pipeline_ready=app_state.pipeline is not None,
        total_session_tokens=app_state.pipeline.total_tokens_used if app_state.pipeline else 0,
        rag_available=RAG_AVAILABLE,
    )

@app.get("/stats", response_model=StatsResponse, tags=["Ops"])
async def stats():
    """Query statistics for the current session."""
    q = app_state.total_queries or 1  # avoid div-by-zero
    return StatsResponse(
        total_queries=app_state.total_queries,
        total_tokens=app_state.pipeline.total_tokens_used if app_state.pipeline else 0,
        avg_latency_ms=app_state.total_latency_ms / q,
        avg_tokens_per_query=(app_state.pipeline.total_tokens_used if app_state.pipeline else 0) / q,
    )

@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat(req: ChatRequest):
    """
    Submit a question to the RAG pipeline.
    Returns the grounded answer, source documents, and token usage.
    """
    if app_state.pipeline is None:
        # Graceful stub response so the API is still testable without full deps
        return ChatResponse(
            answer=(
                f"[STUB MODE — pipeline not initialised]\n\n"
                f"Your question was: '{req.question}'\n\n"
                "In production this endpoint returns a grounded answer retrieved from "
                "your document corpus via FAISS + OpenAI."
            ),
            sources=["stub"],
            tokens_used=0,
            latency_ms=1,
            session_tokens=0,
        )

    t0 = time.monotonic()
    try:
        result = app_state.pipeline.query(req.question)
    except Exception as e:
        logger.error(f"Pipeline query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )

    latency = int((time.monotonic() - t0) * 1000)
    app_state.total_queries += 1
    app_state.total_latency_ms += latency

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        tokens_used=result["tokens_used"],
        latency_ms=latency,
        session_tokens=result["total_tokens"],
    )

@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED, tags=["Management"])
async def ingest(req: IngestRequest):
    """
    Re-ingest documents from the specified directory and rebuild the FAISS index.
    Useful after uploading new documents without restarting the server.
    """
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG dependencies not installed.")
    try:
        pipeline = app_state.pipeline or RAGPipeline(use_local_embeddings=False)
        pipeline.ingest(req.source_dir)
        app_state.pipeline = pipeline
        return {"message": f"Ingest complete from '{req.source_dir}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory", tags=["Management"])
async def reset_memory():
    """Reset conversational memory (start a fresh chat session)."""
    if app_state.pipeline and app_state.pipeline.chain:
        try:
            app_state.pipeline.chain.memory.clear()
            return {"message": "Conversation memory cleared."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"message": "No active pipeline — nothing to reset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
