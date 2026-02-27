"""
Core RAG pipeline — document ingestion, embedding, retrieval, generation.

Design decisions documented in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Lifecycle:
        pipeline = RAGPipeline(config)
        pipeline.initialize()          # load cached index or ingest from scratch
        result = pipeline.query("...")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vectorstore = None
        self.chain = None
        self.total_tokens_used: int = 0

    # ── Lazy imports (keep startup fast; fail clearly if deps missing) ──────
    def _imports(self):
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import ConversationalRetrievalChain
            from langchain.memory import ConversationBufferWindowMemory
            from langchain_community.callbacks.manager import get_openai_callback
            return (PyPDFLoader, TextLoader, DirectoryLoader,
                    RecursiveCharacterTextSplitter,
                    OpenAIEmbeddings, HuggingFaceEmbeddings,
                    FAISS, ChatOpenAI,
                    ConversationalRetrievalChain,
                    ConversationBufferWindowMemory,
                    get_openai_callback)
        except ImportError as e:
            raise ImportError(
                f"LangChain dependencies missing: {e}\n"
                "Run: pip install -r requirements.txt"
            ) from e

    # ── Ingestion ────────────────────────────────────────────────────────────
    def load_documents(self, source_dir: str):
        (PyPDFLoader, TextLoader, DirectoryLoader,
         *_) = self._imports()
        docs = []
        for glob, cls in [("**/*.pdf", PyPDFLoader), ("**/*.txt", TextLoader)]:
            try:
                docs.extend(DirectoryLoader(source_dir, glob=glob, loader_cls=cls).load())
            except Exception as e:
                logger.warning("Loader %s skipped: %s", glob, e)
        logger.info("Loaded %d documents from %s", len(docs), source_dir)
        return docs

    def chunk_documents(self, docs):
        (_, _, _,
         RecursiveCharacterTextSplitter,
         *_) = self._imports()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )
        chunks = splitter.split_documents(docs)
        logger.info("Created %d chunks from %d documents", len(chunks), len(docs))
        return chunks

    # ── Embedding & Vector Store ─────────────────────────────────────────────
    def _get_embeddings(self):
        (_, _, _, _,
         OpenAIEmbeddings, HuggingFaceEmbeddings,
         *_) = self._imports()
        if self.config.use_local_embeddings:
            return HuggingFaceEmbeddings(model_name=self.config.local_embed_model)
        return OpenAIEmbeddings(model=self.config.embed_model)

    def build_vectorstore(self, chunks):
        (_, _, _, _, _, _,
         FAISS, *_) = self._imports()
        embeddings = self._get_embeddings()
        vs = FAISS.from_documents(chunks, embeddings)
        Path(self.config.vector_db_path).parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(self.config.vector_db_path)
        logger.info("FAISS index saved → %s", self.config.vector_db_path)
        return vs

    def load_vectorstore(self):
        (_, _, _, _, _, _,
         FAISS, *_) = self._imports()
        if self.config.vector_db_exists():
            vs = FAISS.load_local(self.config.vector_db_path, self._get_embeddings(), allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded from disk.")
            return vs
        return None

    # ── Chain ────────────────────────────────────────────────────────────────
    def build_chain(self, vectorstore):
        (_, _, _, _, _, _, _,
         ChatOpenAI, ConversationalRetrievalChain,
         ConversationBufferWindowMemory, _) = self._imports()

        retriever = vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.top_k, "fetch_k": self.config.fetch_k},
        )
        llm = ChatOpenAI(
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
            streaming=True,
        )
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=self.config.memory_window_k,
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )

    # ── Public API ───────────────────────────────────────────────────────────
    def ingest(self, source_dir: Optional[str] = None):
        """Full ingest cycle: load → chunk → embed → persist."""
        source_dir = source_dir or self.config.data_dir
        docs = self.load_documents(source_dir)
        if not docs:
            raise ValueError(
                f"No documents found in '{source_dir}'. "
                "Add PDF or TXT files before ingesting."
            )
        chunks = self.chunk_documents(docs)
        self.vectorstore = self.build_vectorstore(chunks)
        self.chain = self.build_chain(self.vectorstore)

    def initialize(self, source_dir: Optional[str] = None):
        """Load cached index if available; otherwise ingest from scratch."""
        cached = self.load_vectorstore()
        if cached:
            self.vectorstore = cached
            self.chain = self.build_chain(cached)
        else:
            self.ingest(source_dir)

    def query(self, question: str) -> dict:
        """
        Run a query. Returns answer, sources, token usage.

        Applies query rewriting before retrieval when the question is
        ambiguous or conversational — see src/rag_chatbot/query_rewriter.py
        for the full decision logic and rationale.
        """
        if self.chain is None:
            raise RuntimeError(
                "Pipeline not initialised. Call initialize() or ingest() first."
            )
        (_, _, _, _, _, _, _, _, _, _,
         get_openai_callback) = self._imports()

        # ── Query rewriting (improves retrieval on ambiguous/short questions) ──
        try:
            from .query_rewriter import rewrite_query
            chat_history = []
            if self.chain and hasattr(self.chain, "memory"):
                chat_history = self.chain.memory.chat_memory.messages
            llm = getattr(self.chain, "combine_docs_chain", None)
            retrieval_question = rewrite_query(question, chat_history, llm)
        except Exception as e:
            logger.warning("Query rewriter import/call failed: %s — using original.", e)
            retrieval_question = question

        with get_openai_callback() as cb:
            # Send rewritten query to retriever but preserve original for generation
            result = self.chain({
                "question": retrieval_question if retrieval_question != question else question
            })
            self.total_tokens_used += cb.total_tokens

        sources = list({
            doc.metadata.get("source", "Unknown")
            for doc in result.get("source_documents", [])
        })
        return {
            "answer": result["answer"],
            "sources": sources,
            "tokens_used": cb.total_tokens,
            "total_tokens": self.total_tokens_used,
            "query_rewritten": retrieval_question != question,
            "retrieval_query": retrieval_question,
        }

    def reset_memory(self):
        """Clear conversational history."""
        if self.chain and hasattr(self.chain, "memory"):
            self.chain.memory.clear()
