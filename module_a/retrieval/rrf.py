"""
module_a/retrieval/rrf.py
──────────────────────────
Reciprocal Rank Fusion (RRF) — merges any number of ranked result lists
into a single unified ranking without score normalisation.

Formula (Cormack et al., 2009):
    RRF_score(d) = Σ_i  1 / (k + rank_i(d))

where:
  k = 60        (constant that dampens the impact of very high-ranked items;
                  empirically optimal across many fusion tasks)
  rank_i(d) = 1-indexed position of document d in result list i
              (documents not appearing in list i contribute 0)

Why RRF is the right choice here:
  • Score-free: BM25 and cosine similarity live on incompatible scales;
    RRF only cares about rank position, so no calibration is needed.
  • Documents appearing in BOTH lists get boosted twice → promotes chunks
    that are both lexically and semantically relevant.
  • Handles lists of different lengths gracefully.
"""

from module_a.config import cfg


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k:            int = None,
) -> list[dict]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        result_lists : Each list contains dicts with at least keys
                       "id", "document", "metadata".
        k            : RRF constant (default: cfg.rrf_k = 60).

    Returns:
        A flat, deduplicated list sorted by RRF score descending.
        Each item has the original keys PLUS "rrf_score".
    """
    k = k or cfg.rrf_k

    rrf_scores: dict[str, float] = {}
    doc_store:  dict[str, dict]  = {}   # id → first-seen full dict

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            # Store the document dict the first time we encounter it
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    # Build final ranked list
    merged: list[dict] = []
    for doc_id in sorted(rrf_scores, key=rrf_scores.get, reverse=True):
        item = {**doc_store[doc_id], "rrf_score": rrf_scores[doc_id]}
        merged.append(item)

    return merged