"""
module_a/retrieval/embedder.py
───────────────────────────────
Singleton wrapper around BAAI/bge-m3.

BGE-M3 is the critical model choice for this system because a single forward
pass produces BOTH:
  • Dense vectors  (1 024-dim) → stored in ChromaDB for semantic search
  • Sparse vectors (lexical weights dict) → available for BGE sparse search
    (we additionally maintain a rank_bm25 index for classic BM25 in bm25_store.py)

Loading a large transformer model is expensive; the singleton pattern ensures
it happens exactly once per process lifetime.
"""

from __future__ import annotations

from typing import Union
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from module_a.config import cfg


class BGE_M3_Embedder:
    """Lazy-loaded singleton wrapper around BAAI/bge-m3."""

    _instance: "BGE_M3_Embedder | None" = None

    def __init__(self) -> None:
        print(f"[Embedder] Loading {cfg.embedding_model} on device='{cfg.embedding_device}'…")
        self.model = BGEM3FlagModel(
            cfg.embedding_model,
            use_fp16=(cfg.embedding_device != "cpu"),  # FP16 only on GPU
        )
        print("[Embedder] ✓ Model ready.")

    # ── Singleton accessor ────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "BGE_M3_Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        texts:      Union[str, list[str]],
        batch_size: int = 12,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Full encode: returns dense vectors AND sparse lexical weights.

        Returns:
            dense_vecs     : np.ndarray of shape (N, 1024)
            lexical_weights: list[dict]  —  token_id (str) → float weight
        """
        if isinstance(texts, str):
            texts = [texts]
        outputs = self.model.encode(
            texts,
            batch_size         = batch_size,
            max_length         = 8192,
            return_dense       = True,
            return_sparse      = True,
            return_colbert_vecs= False,
        )
        return outputs["dense_vecs"], outputs["lexical_weights"]

    def encode_dense_only(
        self,
        texts:      Union[str, list[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Lightweight path used at query time — skips sparse computation
        when we only need the dense vector for ChromaDB lookup.

        Returns:
            np.ndarray of shape (N, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]
        outputs = self.model.encode(
            texts,
            batch_size         = batch_size,
            max_length         = 512,   # Shorter queries → smaller window
            return_dense       = True,
            return_sparse      = False,
            return_colbert_vecs= False,
        )
        return outputs["dense_vecs"]