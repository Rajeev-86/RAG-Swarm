"""
module_a/retrieval/hybrid_search.py
─────────────────────────────────────
Stage 2 + 3 of the RAG pipeline: Hybrid Search → RRF.

Both search arms run concurrently (ThreadPoolExecutor) to minimise latency:
  ┌─────────────────────┐   ┌─────────────────────┐
  │  Dense Search       │   │  BM25 Keyword Search │
  │  (ChromaDB cosine)  │   │  (rank_bm25 index)   │
  └────────┬────────────┘   └──────────┬───────────┘
           │                           │
           └──────────── RRF ──────────┘
                          │
                   Merged ranked list
"""

import concurrent.futures
from typing import Optional
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
from module_a.config import cfg
from module_a.retrieval.embedder import BGE_M3_Embedder
from module_a.retrieval.vector_store import ChromaVectorStore
from module_a.retrieval.bm25_store import BM25Store
from module_a.retrieval.rrf import reciprocal_rank_fusion


class HybridSearchEngine:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_store:   BM25Store,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store   = bm25_store
        self.embedder     = BGE_M3_Embedder.get()

    def search(
        self,
        query:         str,
        top_k:         int           = None,
        domain_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Execute dense and BM25 searches in parallel then fuse with RRF.

        Args:
            query         : Natural language query string.
            top_k         : Candidates per arm; total pool before reranking
                            may be up to 2 × top_k (some overlap expected).
            domain_filter : If set, restrict both arms to this domain only.

        Returns:
            RRF-merged list, sorted by descending rrf_score.
        """
        top_k = top_k or cfg.top_k_retrieval

        def _dense():
            vec = self.embedder.encode_dense_only(query)
            return self.vector_store.query(
                query_embedding = vec[0],
                top_k           = top_k,
                domain_filter   = domain_filter,
            )

        def _sparse():
            return self.bm25_store.query(
                query_text    = query,
                top_k         = top_k,
                domain_filter = domain_filter,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_dense  = pool.submit(_dense)
            future_sparse = pool.submit(_sparse)
            dense_results  = future_dense.result()
            sparse_results = future_sparse.result()

        merged = reciprocal_rank_fusion([dense_results, sparse_results])
        return merged[:top_k]
