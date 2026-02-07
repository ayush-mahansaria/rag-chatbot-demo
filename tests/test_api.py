"""
API smoke tests — no pipeline initialised, validates stub-mode responses
and endpoint contract so CI passes without an OpenAI key.
"""
import pytest
from fastapi.testclient import TestClient

# Patch RAGPipeline import before importing api
import sys
from unittest.mock import MagicMock, patch

# Stub out langchain/openai so import succeeds in CI
sys.modules.setdefault("langchain", MagicMock())
sys.modules.setdefault("langchain.document_loaders", MagicMock())
sys.modules.setdefault("langchain.text_splitter", MagicMock())
sys.modules.setdefault("langchain.embeddings", MagicMock())
sys.modules.setdefault("langchain.vectorstores", MagicMock())
sys.modules.setdefault("langchain.chat_models", MagicMock())
sys.modules.setdefault("langchain.chains", MagicMock())
sys.modules.setdefault("langchain.memory", MagicMock())
sys.modules.setdefault("langchain.callbacks", MagicMock())
sys.modules.setdefault("langchain.schema", MagicMock())

from api import app   # noqa: E402

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "pipeline_ready" in data
        assert "total_session_tokens" in data
        assert data["status"] == "ok"


class TestStatsEndpoint:
    def test_stats_returns_200(self):
        r = client.get("/stats")
        assert r.status_code == 200

    def test_stats_schema(self):
        data = client.get("/stats").json()
        for key in ["total_queries", "total_tokens", "avg_latency_ms", "avg_tokens_per_query"]:
            assert key in data


class TestChatEndpoint:
    def test_chat_stub_response(self):
        r = client.post("/chat", json={"question": "What is RAG?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "latency_ms" in data

    def test_chat_empty_question_rejected(self):
        r = client.post("/chat", json={"question": ""})
        assert r.status_code == 422  # Pydantic validation

    def test_chat_too_long_question_rejected(self):
        r = client.post("/chat", json={"question": "x" * 2001})
        assert r.status_code == 422

    def test_chat_answer_contains_question(self):
        q = "What does retrieval augmented mean?"
        data = client.post("/chat", json={"question": q}).json()
        assert q in data["answer"] or len(data["answer"]) > 0


class TestMemoryEndpoint:
    def test_reset_memory_no_pipeline(self):
        r = client.delete("/memory")
        assert r.status_code == 200
        assert "message" in r.json()
