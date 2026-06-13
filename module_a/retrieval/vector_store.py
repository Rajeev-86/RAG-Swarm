"""
module_a/retrieval/vector_store.py
────────────────────────────────────
ChromaDB wrapper for dense-vector storage and semantic (cosine) search.

Design notes:
  • Uses cosine distance space ("hnsw:space": "cosine").
  • Batches add() calls to stay under ChromaDB's internal 5 461-item limit.
  • Domain metadata stored on every chunk → enables domain-filtered queries
    when a specific agent only wants its own documents.
  • Distance is converted to similarity (1 – distance) so scores are
    intuitive: 1.0 = identical, 0.0 = orthogonal.
"""

import os
import numpy as np
import chromadb
from chromadb.config import Settings

from module_a.config import cfg
from module_a.ingestion.chunker import DocumentChunk

_BATCH_LIMIT = 1_000   # Stay well under ChromaDB's hard limit


class ChromaVectorStore:
    def __init__(self) -> None:
        os.makedirs(cfg.chroma_persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path    = cfg.chroma_persist_dir,
            settings= Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name    = cfg.chroma_collection_name,
            metadata= {"hnsw:space": "cosine"},
        )
        print(
            f"[ChromaDB] ✓ Collection '{cfg.chroma_collection_name}' ready. "
            f"Stored chunks: {self.collection.count()}"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks:           list[DocumentChunk],
        dense_embeddings: np.ndarray,
    ) -> None:
        """
        Persist chunks with their dense embeddings.
        Skips IDs that already exist to make ingestion idempotent.
        """
        if len(chunks) != len(dense_embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) ≠ embedding count "
                f"({len(dense_embeddings)})"
            )

        # Filter out already-stored chunks
        existing_ids: set[str] = set()
        try:
            existing = self.collection.get(ids=[c.chunk_id for c in chunks])
            existing_ids = set(existing["ids"])
        except Exception:
            pass  # collection.get() with missing ids raises in some versions

        new_chunks = [(c, e) for c, e in zip(chunks, dense_embeddings)
                      if c.chunk_id not in existing_ids]
        if not new_chunks:
            print("[ChromaDB] All chunks already indexed — skipping.")
            return

        for start in range(0, len(new_chunks), _BATCH_LIMIT):
            batch = new_chunks[start : start + _BATCH_LIMIT]
            self.collection.add(
                ids        = [c.chunk_id       for c, _ in batch],
                documents  = [c.content        for c, _ in batch],
                embeddings = [e.tolist()        for _, e in batch],
                metadatas  = [c.metadata        for c, _ in batch],
            )

        print(
            f"[ChromaDB] ✓ Added {len(new_chunks)} new chunks. "
            f"Total stored: {self.collection.count()}"
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k:           int  = None,
        domain_filter:   str  = None,
    ) -> list[dict]:
        """
        Semantic nearest-neighbour search.

        Args:
            query_embedding : 1-D numpy array of shape (1024,)
            top_k           : number of results to return
            domain_filter   : if set, restrict to chunks whose domain matches

        Returns:
            List of dicts with keys: id, document, metadata, score
        """
        top_k = top_k or cfg.top_k_retrieval
        where = {"domain": {"$eq": domain_filter}} if domain_filter else None

        results = self.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = min(top_k, self.collection.count() or 1),
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append({
                "id":       doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1.0 - results["distances"][0][i],  # cosine sim
            })
        return hits

    def count(self) -> int:
        return self.collection.count()