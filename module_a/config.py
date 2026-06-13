"""
module_a/config.py
─────────────────
Central configuration for Module A.
All tunable parameters live here — never hard-coded in downstream modules.
Reads from .env automatically via python-dotenv.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


class LLMBackend(str, Enum):
    GROQ   = "groq"
    OLLAMA = "ollama"


@dataclass
class RAGConfig:
    # ── LLM Backend ──────────────────────────────────────────────────────────
    llm_backend: LLMBackend = LLMBackend(os.getenv("LLM_BACKEND", "groq"))

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str  = os.getenv("GROQ_API_KEY", "")
    groq_model:   str  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model:    str = os.getenv("OLLAMA_MODEL", "llama3:8b")

    # ── Embedding & Reranking models ─────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    reranker_model:  str = "BAAI/bge-reranker-v2-m3"
    # "cpu" always works; switch to "cuda" if you have a GPU
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir:     str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    chroma_collection_name: str = "ma_data_room"

    # ── BM25 Index ───────────────────────────────────────────────────────────
    bm25_index_path: str = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size:    int = 512   # Target word count per chunk
    chunk_overlap: int = 64    # Overlap in words between consecutive chunks

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = 20  # Candidates retrieved per search arm before RRF
    top_k_reranked:  int = 5   # Final chunks after cross-encoder reranking
    rrf_k:           int = 60  # RRF constant (standard value from the paper)

    # ── Query Decomposition ──────────────────────────────────────────────────
    max_subqueries: int = 4    # Maximum sub-queries produced from one complex query

    # ── M&A Domains ──────────────────────────────────────────────────────────
    domains: list = field(default_factory=lambda: [
        "legal", "financial", "hr", "cybersecurity"
    ])


# ── Singleton ────────────────────────────────────────────────────────────────
cfg = RAGConfig()