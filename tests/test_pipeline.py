"""
Unit tests for the RAG pipeline (no OpenAI calls — uses local HuggingFace embeddings).
Run: pytest tests/ -v
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Point to local embeddings so tests run without an API key
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app import RAGPipeline, CHUNK_SIZE, CHUNK_OVERLAP


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_docs(tmp_path_factory):
    """Write a small corpus to a temp dir and return the path."""
    d = tmp_path_factory.mktemp("data")
    (d / "doc1.txt").write_text(
        "LangChain is a framework for building LLM-powered applications.\n"
        "It supports RAG pipelines, agents, and tool use.\n" * 20
    )
    (d / "doc2.txt").write_text(
        "FAISS (Facebook AI Similarity Search) enables fast nearest-neighbour lookup.\n"
        "It is widely used for vector search in production RAG systems.\n" * 20
    )
    return str(d)


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline(use_local_embeddings=True)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────
class TestDocumentLoading:
    def test_load_txt_files(self, pipeline, sample_docs):
        docs = pipeline.load_documents(sample_docs)
        assert len(docs) >= 2, "Should load at least 2 documents"

    def test_empty_dir_returns_empty(self, pipeline, tmp_path):
        docs = pipeline.load_documents(str(tmp_path))
        assert docs == []


class TestChunking:
    def test_chunk_size_respected(self, pipeline, sample_docs):
        docs = pipeline.load_documents(sample_docs)
        chunks = pipeline.chunk_documents(docs)
        oversized = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.2]
        assert len(oversized) == 0, "Chunks should not vastly exceed CHUNK_SIZE"

    def test_overlap_produces_more_chunks(self, pipeline, sample_docs):
        docs = pipeline.load_documents(sample_docs)
        chunks = pipeline.chunk_documents(docs)
        # More chunks than raw docs means splitting occurred
        assert len(chunks) > len(docs)

    def test_chunks_have_metadata(self, pipeline, sample_docs):
        docs = pipeline.load_documents(sample_docs)
        chunks = pipeline.chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata


class TestVectorStore:
    def test_build_and_similarity_search(self, pipeline, sample_docs, tmp_path):
        docs = pipeline.load_documents(sample_docs)
        chunks = pipeline.chunk_documents(docs)
        vs = pipeline.build_vectorstore(chunks)
        results = vs.similarity_search("What is LangChain?", k=3)
        assert len(results) == 3
        assert any("LangChain" in r.page_content for r in results)

    def test_vectorstore_persistence(self, pipeline, sample_docs, tmp_path):
        """Index saved to disk and reloaded produces same results."""
        import app as app_module
        original_path = app_module.VECTOR_DB_PATH
        app_module.VECTOR_DB_PATH = str(tmp_path / "test_index")

        docs = pipeline.load_documents(sample_docs)
        chunks = pipeline.chunk_documents(docs)
        vs1 = pipeline.build_vectorstore(chunks)
        r1 = vs1.similarity_search("FAISS vector search", k=2)

        vs2 = pipeline.load_vectorstore()
        assert vs2 is not None
        r2 = vs2.similarity_search("FAISS vector search", k=2)
        assert r1[0].page_content == r2[0].page_content

        app_module.VECTOR_DB_PATH = original_path  # restore


class TestQueryPipeline:
    def test_query_returns_expected_keys(self, pipeline, sample_docs):
        """Mock the LLM to avoid real API calls; verify response structure."""
        pipeline.initialize(sample_docs)

        mock_chain = MagicMock()
        mock_chain.return_value = {
            "answer": "LangChain is a framework for building LLM applications.",
            "source_documents": [],
        }
        pipeline.chain = mock_chain

        result = pipeline.query("What is LangChain?")
        assert "answer" in result
        assert "sources" in result
        assert "tokens_used" in result
        assert isinstance(result["sources"], list)

    def test_uninitialized_pipeline_raises(self):
        fresh = RAGPipeline(use_local_embeddings=True)
        with pytest.raises(RuntimeError, match="not initialized"):
            fresh.query("hello")
