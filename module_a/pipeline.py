"""
module_a/pipeline.py
─────────────────────
RAGPipeline — the single entry point for Module A.

Orchestrates all 5 stages as described in the spec:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 1  │  Query Decomposition (LLM)                              │
  │  Stage 2  │  Hybrid Search — BM25 + Dense, per sub-query, parallel  │
  │  Stage 3  │  Reciprocal Rank Fusion (inside HybridSearchEngine)     │
  │  Stage 4  │  Cross-Encoder Reranking (bge-reranker-v2-m3)           │
  │  Stage 5  │  Context Assembly & formatting for agent injection       │
  └─────────────────────────────────────────────────────────────────────┘

Usage:

    pipeline = RAGPipeline()
    pipeline.ingest("./data/data_room")      # one-time; idempotent on re-run

    result = pipeline.retrieve(
        "What change-of-control clauses exist and what is the revenue breakdown?"
    )
    context_string = pipeline.format_context(result)
    # Pass context_string into the agent's system/user prompt

RetrievalResult is a plain dataclass — safe to pickle for inter-agent transfer
in Module C's mesh layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
from module_a.config import cfg
from module_a.ingestion.loader import load_data_room
from module_a.ingestion.chunker import chunk_documents
from module_a.retrieval.embedder import BGE_M3_Embedder
from module_a.retrieval.vector_store import ChromaVectorStore
from module_a.retrieval.bm25_store import BM25Store
from module_a.retrieval.hybrid_search import HybridSearchEngine
from module_a.reranker import CrossEncoderReranker
from module_a.query_decomposer import decompose_query


# ── Result Type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query:             str
    subqueries:        list[dict]                    # [{subquery, domain}, …]
    retrieved_chunks:  list[dict]                    # Reranked, top-K chunks
    domain_breakdown:  dict[str, list[dict]] = field(default_factory=dict)
    # {domain: [chunk, …]}  — useful for per-agent context slicing


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Initialise once per process; call ingest() when the data room changes,
    then retrieve() for every agent query.
    """

    def __init__(self) -> None:
        self.embedder      = BGE_M3_Embedder.get()
        self.vector_store  = ChromaVectorStore()
        self.bm25_store    = BM25Store()
        self.hybrid_engine = HybridSearchEngine(self.vector_store, self.bm25_store)
        self.reranker      = CrossEncoderReranker.get()
        print("[RAGPipeline] ✓ All components initialised.\n")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, data_room_dir: str) -> None:
        """
        Load → Chunk → Embed → Index all documents in data_room_dir.
        Safe to call multiple times; already-indexed chunks are skipped.
        """
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  [RAGPipeline] Ingesting: {data_room_dir}")
        print(f"{sep}\n")

        raw_docs = load_data_room(data_room_dir)
        if not raw_docs:
            print("[RAGPipeline] ⚠ No documents found. Aborting.")
            return

        chunks = chunk_documents(raw_docs)

        print(f"\n[RAGPipeline] Embedding {len(chunks)} chunks…")
        texts      = [c.content for c in chunks]
        dense_vecs, _ = self.embedder.encode(texts)   # sparse unused at index time

        self.vector_store.add_chunks(chunks, dense_vecs)
        self.bm25_store.add_chunks(chunks)

        print(f"\n{sep}")
        print(f"  [RAGPipeline] ✓ Ingestion complete — {len(chunks)} chunks indexed.")
        print(f"{sep}\n")

    # ── Retrieval (5 stages) ──────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        domain_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Run the full 5-stage pipeline for a single query string.

        Args:
            query         : The user / agent query.
            domain_filter : Optional hard override — restrict every sub-query
                            to this domain (useful when called by a domain agent
                            that only wants its own documents).
        Returns:
            RetrievalResult with the top-K reranked chunks and metadata.
        """
        # ── Stage 1: Query Decomposition ──────────────────────────────────────
        print("\n── Stage 1 ▸ Query Decomposition ─────────────────────────────")
        subqueries = decompose_query(query)
        for sq in subqueries:
            print(f"   [{sq['domain']:>14}] {sq['subquery']}")

        # ── Stage 2 + 3: Hybrid Search + RRF per sub-query ───────────────────
        print("\n── Stage 2+3 ▸ Hybrid Search + RRF ───────────────────────────")
        candidate_pool: dict[str, dict] = {}   # chunk_id → dict (deduplicated)

        for sq in subqueries:
            sub_domain = sq["domain"] if sq["domain"] != "unknown" else domain_filter
            hits = self.hybrid_engine.search(
                query         = sq["subquery"],
                domain_filter = sub_domain,
            )
            for hit in hits:
                cid = hit["id"]
                # Keep the hit with the highest RRF score on collision
                if cid not in candidate_pool or \
                   hit.get("rrf_score", 0) > candidate_pool[cid].get("rrf_score", 0):
                    candidate_pool[cid] = hit

        candidates = list(candidate_pool.values())
        print(f"   Candidate pool after dedup: {len(candidates)} chunks")

        # ── Stage 4: Cross-Encoder Reranking ──────────────────────────────────
        print("\n── Stage 4 ▸ Cross-Encoder Reranking ─────────────────────────")
        reranked = self.reranker.rerank(query, candidates)
        for i, c in enumerate(reranked, 1):
            fname = c.get("metadata", {}).get("filename", "?")
            print(f"   #{i}  score={c['rerank_score']:.4f}  [{c['metadata'].get('domain','?')}]  {fname}")

        # ── Stage 5: Context Assembly ─────────────────────────────────────────
        print("\n── Stage 5 ▸ Context Assembly ─────────────────────────────────")
        domain_breakdown: dict[str, list[dict]] = {}
        for chunk in reranked:
            d = chunk.get("metadata", {}).get("domain", "unknown")
            domain_breakdown.setdefault(d, []).append(chunk)
        print(f"   Domains represented: {list(domain_breakdown.keys())}")

        return RetrievalResult(
            query            = query,
            subqueries       = subqueries,
            retrieved_chunks = reranked,
            domain_breakdown = domain_breakdown,
        )

    # ── Context Formatter ────────────────────────────────────────────────────

    def format_context(self, result: RetrievalResult) -> str:
        """
        Render a RetrievalResult into a clean, numbered context block
        ready to inject into an agent's prompt.

        Example output:
            [Context for: "What change-of-control clauses exist?"]

            [Chunk 1 | Domain: legal | Source: agreement.pdf | Score: 0.9341]
            In the event of a Change of Control, all outstanding options…
            …
        """
        lines = [f'[Context for: "{result.query}"]\n']
        for i, chunk in enumerate(result.retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            lines.append(
                f"[Chunk {i} | Domain: {meta.get('domain','?')} | "
                f"Source: {meta.get('filename','?')} | "
                f"Score: {chunk.get('rerank_score', 0):.4f}]\n"
                f"{chunk['document']}\n"
            )
        return "\n".join(lines)
