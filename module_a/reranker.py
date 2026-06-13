"""
module_a/reranker.py
─────────────────────
Stage 4: Cross-Encoder Reranking using BAAI/bge-reranker-v2-m3.

Why a second ranking pass?
  Bi-encoder models (BGE-M3) compare query and document embeddings
  independently — fast, but approximate. A cross-encoder sees the full
  (query, document) pair jointly, which yields much higher precision at the
  cost of speed. Running it only on the top-K RRF candidates (not the entire
  corpus) keeps latency manageable.

The spec cites Context Precision as a key metric:
  "Did the Cross-Encoder rank the exact necessary clause at rank #1?"
  → This module directly drives that metric.
"""

from __future__ import annotations

from FlagEmbedding import FlagReranker
from module_a.config import cfg


class CrossEncoderReranker:
    """Lazy-loaded singleton wrapper around BAAI/bge-reranker-v2-m3."""

    _instance: "CrossEncoderReranker | None" = None

    def __init__(self) -> None:
        print(f"[Reranker] Loading {cfg.reranker_model}…")
        self.model = FlagReranker(
            cfg.reranker_model,
            use_fp16=(cfg.embedding_device != "cpu"),
        )
        print("[Reranker] ✓ Model ready.")

    @classmethod
    def get(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = None,
    ) -> list[dict]:
        """
        Re-score and re-rank a list of candidate chunks.

        Args:
            query      : The original (or primary) query string.
            candidates : List of dicts that each contain a "document" key.
            top_k      : How many top results to return.

        Returns:
            Candidates sorted by rerank_score descending, truncated to top_k.
            The "rerank_score" key (0–1 normalised) is added in-place.
        """
        top_k = top_k or cfg.top_k_reranked
        if not candidates:
            return []

        pairs  = [[query, c["document"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]