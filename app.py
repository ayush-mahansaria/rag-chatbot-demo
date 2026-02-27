"""
RAG Chatbot Demo — Ayush Mahansaria
Retrieval-Augmented Generation pipeline using LangChain, FAISS, and OpenAI.
Demonstrates end-to-end GenAI solution design from document ingestion to chat.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE = 1000                   # Can be larger because gpt-4o-mini handles context better
TOP_K_RETRIEVAL = 8                 # Retrieve more chunks for better accuracy
CHUNK_OVERLAP = 150
MODEL_NAME = "gpt-4o-mini"          # swap to "gpt-5-mini" for higher quality
EMBED_MODEL = "text-embedding-3-large" # Much more accurate retrieval than ada-002
VECTOR_DB_PATH = "vectorstore/faiss_index"


# ─────────────────────────────────────────────
# Core RAG Pipeline
# ─────────────────────────────────────────────
class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Document ingestion & chunking
      2. Embedding & FAISS vector store
      3. Conversational retrieval chain with memory
    """

    def __init__(self, use_local_embeddings: bool = False):
        self.use_local_embeddings = use_local_embeddings
        self.vectorstore: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.total_tokens_used = 0

    # ---------- Ingestion ----------
    def load_documents(self, source_dir: str = "data/") -> List[Document]:
        """Load PDFs and .txt files from a directory."""
        loaders = {
            "**/*.pdf": PyPDFLoader,
            "**/*.txt": TextLoader,
        }
        docs: List[Document] = []
        for glob, loader_cls in loaders.items():
            try:
                loader = DirectoryLoader(source_dir, glob=glob, loader_cls=loader_cls)
                docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Loader {glob} skipped: {e}")
        logger.info(f"Loaded {len(docs)} raw documents.")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks for retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents.")
        return chunks

    # ---------- Embedding & Vector Store ----------
    def build_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Embed chunks and persist a FAISS index."""
        embeddings = self._get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        Path(VECTOR_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(VECTOR_DB_PATH)
        logger.info(f"FAISS index saved → {VECTOR_DB_PATH}")
        return vectorstore

    def load_vectorstore(self) -> Optional[FAISS]:
        """Load a persisted FAISS index if available."""
        if Path(f"{VECTOR_DB_PATH}.faiss").exists():
            embeddings = self._get_embeddings()
            vs = FAISS.load_local(VECTOR_DB_PATH, embeddings)
            logger.info("FAISS index loaded from disk.")
            return vs
        return None

    def _get_embeddings(self):
        if self.use_local_embeddings:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return OpenAIEmbeddings(model=EMBED_MODEL)

    # ---------- Chain ----------
    def build_chain(self, vectorstore: FAISS) -> ConversationalRetrievalChain:
        """Wire retriever + LLM into a conversational chain with sliding-window memory."""
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": 20},
        )
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2, streaming=True)
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=6,  # keep last 6 turns
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        return chain

    # ---------- Public API ----------
    def ingest(self, source_dir: str = "data/"):
        """Full ingest: load → chunk → embed → save."""
        docs = self.load_documents(source_dir)
        if not docs:
            raise ValueError(f"No documents found in '{source_dir}'. Add PDF or TXT files.")
        chunks = self.chunk_documents(docs)
        self.vectorstore = self.build_vectorstore(chunks)
        self.chain = self.build_chain(self.vectorstore)

    def initialize(self, source_dir: str = "data/"):
        """Try to load cached vectorstore; otherwise ingest from scratch."""
        cached = self.load_vectorstore()
        if cached:
            self.vectorstore = cached
            self.chain = self.build_chain(cached)
        else:
            self.ingest(source_dir)

    def query(self, question: str) -> dict:
        """Run a query and return answer + sources + token usage."""
        if self.chain is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        with get_openai_callback() as cb:
            result = self.chain({"question": question})
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
        }


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
def build_ui():
    st.set_page_config(
        page_title="RAG Chatbot Demo",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 RAG Chatbot Demo")
    st.caption("Retrieval-Augmented Generation · LangChain · FAISS · OpenAI")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        use_local = st.toggle("Use local embeddings (HuggingFace)", value=False)
        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload documents (PDF / TXT)",
            accept_multiple_files=True,
            type=["pdf", "txt"],
        )
        if uploaded:
            Path("data/").mkdir(exist_ok=True)
            for f in uploaded:
                (Path("data/") / f.name).write_bytes(f.read())
            st.success(f"Uploaded {len(uploaded)} file(s) to data/")
        st.markdown("---")
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.session_state.pop("pipeline", None)
            st.session_state.pop("messages", None)
            st.rerun()
        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "Built with LangChain · FAISS · Streamlit  \n"
            "Demonstrates: RAG, MMR retrieval, sliding-window memory, token tracking."
        )

    # ── Pipeline init ──
    if "pipeline" not in st.session_state:
        with st.spinner("Initialising RAG pipeline…"):
            try:
                pipeline = RAGPipeline(use_local_embeddings=use_local)
                Path("data/").mkdir(exist_ok=True)
                # seed a demo doc if data/ is empty
                demo_path = Path("data/demo.txt")
                if not any(Path("data/").iterdir()):
                    demo_path.write_text(
                        "This is a RAG Chatbot demo built by Ayush Mahansaria.\n"
                        "The system uses LangChain for orchestration, FAISS for vector search, "
                        "and OpenAI for language generation.\n"
                        "Ask me anything about this demo or upload your own documents!"
                    )
                pipeline.initialize()
                st.session_state["pipeline"] = pipeline
                st.success("✅ Pipeline ready.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.info("Make sure OPENAI_API_KEY is set in your .env file.")
                st.stop()

    pipeline: RAGPipeline = st.session_state["pipeline"]

    # ── Chat history ──
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")

    # ── Input ──
    if prompt := st.chat_input("Ask a question about your documents…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t0 = time.time()
                result = pipeline.query(prompt)
                elapsed = time.time() - t0

            st.markdown(result["answer"])
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱ Latency", f"{elapsed:.2f}s")
            col2.metric("🔤 Tokens (this call)", result["tokens_used"])
            col3.metric("🔤 Tokens (session)", result["total_tokens"])

            if result["sources"]:
                with st.expander("📚 Retrieved Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- `{s}`")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


if __name__ == "__main__":
    build_ui()
