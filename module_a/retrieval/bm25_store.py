"""
module_a/retrieval/bm25_store.py
─────────────────────────────────
BM25Okapi keyword index built with rank_bm25.

Why keep BM25 alongside dense vectors?
  • Dense search excels at semantic paraphrasing ("termination fee" ↔ "break-up
    payment") but misses exact term matches for proper nouns, code identifiers,
    CVE numbers, clause references (§ 4.2(b)), etc.
  • BM25 is the precise complement: it fires on exact tokens.
  • RRF then fuses both signals without requiring score normalisation.

Persistence:
  The entire index (tokenised corpus + chunk objects) is pickled to disk so
  re-ingestion of unchanged documents is idempotent.
"""

import os
import pickle
from rank_bm25 import BM25Okapi

from module_a.config import cfg
from module_a.ingestion.chunker import DocumentChunk


def _tokenise(text: str) -> list[str]:
    """
    Lowercase tokeniser that strips punctuation and removes tokens shorter
    than 3 characters to reduce noise from articles and prepositions.
    """
    import re
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return [t for t in tokens if len(t) >= 3]


class BM25Store:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._corpus: list[list[str]]     = []   # Tokenised documents
        self._bm25:   BM25Okapi | None    = None
        self._load_if_exists()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if os.path.exists(cfg.bm25_index_path):
            print(f"[BM25] Loading existing index from '{cfg.bm25_index_path}'…")
            with open(cfg.bm25_index_path, "rb") as fh:
                data = pickle.load(fh)
            self._chunks = data["chunks"]
            self._corpus = data["corpus"]
            self._bm25   = BM25Okapi(self._corpus)
            print(f"[BM25] ✓ Loaded {len(self._chunks)} documents from disk.")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(cfg.bm25_index_path) or ".", exist_ok=True)
        with open(cfg.bm25_index_path, "wb") as fh:
            pickle.dump({"chunks": self._chunks, "corpus": self._corpus}, fh)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Append new chunks to the index. Skips chunks already present by chunk_id
        to keep ingestion idempotent.
        """
        existing_ids = {c.chunk_id for c in self._chunks}
        new_chunks   = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            print("[BM25] All chunks already indexed — skipping.")
            return

        new_corpus = [_tokenise(c.content) for c in new_chunks]
        self._chunks.extend(new_chunks)
        self._corpus.extend(new_corpus)
        self._bm25 = BM25Okapi(self._corpus)
        self._save()
        print(f"[BM25] ✓ Index updated. Total documents: {len(self._chunks)}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text:    str,
        top_k:         int = None,
        domain_filter: str = None,
    ) -> list[dict]:
        """
        BM25 keyword search.

        Returns:
            List of dicts with keys: id, document, metadata, score
            Sorted by BM25 score descending.
        """
        if self._bm25 is None or not self._chunks:
            return []

        top_k          = top_k or cfg.top_k_retrieval
        tokenised_query = _tokenise(query_text)
        scores          = self._bm25.get_scores(tokenised_query)

        # Build (index, score) pairs, filtering by domain if requested
        pairs = [
            (i, float(score))
            for i, (score, chunk) in enumerate(zip(scores, self._chunks))
            if score > 0
            and (domain_filter is None or chunk.domain == domain_filter)
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)

        hits: list[dict] = []
        for idx, score in pairs[:top_k]:
            chunk = self._chunks[idx]
            hits.append({
                "id":       chunk.chunk_id,
                "document": chunk.content,
                "metadata": chunk.metadata,
                "score":    score,
            })
        return hits

    def __len__(self) -> int:
        return len(self._chunks)