"""
module_b/bridge.py
───────────────────
Connects Module B (domain agents) to Module D / Module C (leader + mesh).

Converts PeerQuery objects from AgentResult into AuditTask objects that
Module D dispatches to the Module C mesh for P2P debate.

The complete pipeline this enables:

    Module A  RAGPipeline.retrieve()
        → RetrievalResult

    Module B  run_all_agents(retrieval_result)
        → dict[DomainType, AgentResult]

    Bridge    agent_results_to_audit_tasks(results)       ← THIS MODULE
        → List[AuditTask]

    Module D  build_leader_graph().invoke(
                  make_initial_global_state(audit_tasks, run_id))
        → dispatches each AuditTask to Module C mesh
        → checkpoints / kill-switch on runaway debates
        → GlobalState with shared_memory, audit_log

Key design decisions
──────────────────────
Deduplication by (source, target) pair:
    Module C's kill-switch fires on "context fragmentation" — defined as
    the same agent pair appearing in >1 concurrent unresolved thread.
    Merging multiple Financial→Legal PeerQueries into one AuditTask prevents
    this from triggering the moment the mesh starts.

Urgency-first ordering:
    AuditTasks are sorted so CRITICAL / HIGH urgency queries run first.
    If a kill-switch fires mid-run and truncates the queue, the highest-risk
    debates will have completed — correct triage behaviour.

Evidence enrichment:
    PeerQuery.context_snippet is capped at 500 chars by BaseAgent.
    The bridge looks up the originating Finding in AgentResult to include
    the full description + evidence_quote, giving mesh agents much richer
    context than the snippet alone.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from module_b.schemas import AgentResult, DomainType, Finding, PeerQuery, RiskLevel
from module_d.leader_state import AuditTask, make_audit_task

logger = logging.getLogger(__name__)

_URGENCY_WEIGHT: Dict[RiskLevel, int] = {
    RiskLevel.CRITICAL: 4,
    RiskLevel.HIGH:     3,
    RiskLevel.MEDIUM:   2,
    RiskLevel.LOW:      1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finding_matches_query(finding: Finding, peer_query: PeerQuery) -> bool:
    """
    Return True if `finding` is the source of `peer_query`.

    Two cases to handle:
      • finding.peer_review_question set explicitly → compare directly.
      • peer_review_question is None → BaseAgent._compile_peer_queries() used
        the default string f"Please cross-validate this finding: {finding.title}",
        which was written into peer_query.question.
    """
    if not finding.requires_peer_review:
        return False
    if finding.peer_review_target != peer_query.target_domain:
        return False
    if finding.peer_review_question is not None:
        return finding.peer_review_question == peer_query.question
    # Default question format from BaseAgent._compile_peer_queries()
    return f"Please cross-validate this finding: {finding.title}" == peer_query.question


def _build_evidence_chunks(
    peer_query: PeerQuery,
    source_result: Optional[AgentResult] = None,
) -> List[str]:
    """
    Assemble evidence chunks for a single PeerQuery.

    Priority:
      1. Full finding description + verbatim evidence_quote from AgentResult.
         This is always richer than the truncated context_snippet.
      2. The raw 500-char context_snippet on PeerQuery (fallback when no
         AgentResult is provided or no matching finding is found).

    Returns an empty list only when both sources are empty — the caller
    logs a warning in that case so the operator knows the debate is under-fed.
    """
    chunks: List[str] = []

    if source_result:
        for finding in source_result.findings:
            if _finding_matches_query(finding, peer_query):
                if finding.description:
                    chunks.append(
                        f"[{peer_query.source_domain.value.upper()} FINDING"
                        f" — {finding.title}]\n"
                        f"Risk: {finding.risk_level.value}\n"
                        f"{finding.description}"
                    )
                if finding.evidence_quote:
                    chunks.append(
                        f"[SOURCE QUOTE — {finding.domain.value}]\n"
                        f"{finding.evidence_quote}"
                    )
                break  # One finding maps to one PeerQuery

    # Fallback: use context_snippet when no richer source was found
    if not chunks and peer_query.context_snippet:
        chunks.append(
            f"[{peer_query.source_domain.value.upper()} CONTEXT]\n"
            f"{peer_query.context_snippet}"
        )

    return chunks


def _merge_by_pair(
    queries: List[tuple[PeerQuery, AgentResult]],
) -> List[AuditTask]:
    """
    Group queries by (source_domain, target_domain) and emit one AuditTask
    per unique pair.

    Assumes `queries` is pre-sorted by urgency descending (done by the public
    caller before calling this). The first query in each group is therefore
    the most urgent and becomes the task's "lead" — its query_id appears in
    the task_id for traceability.

    Single-query groups → direct conversion via peer_query_to_audit_task.
    Multi-query groups → questions numbered, evidence merged and deduplicated.
    """
    # defaultdict preserves insertion order (Python 3.7+), so groups are
    # ordered by the first (most urgent) query encountered for each pair.
    groups: dict[
        tuple[str, str],
        list[tuple[PeerQuery, AgentResult]]
    ] = defaultdict(list)

    for pq, src in queries:
        groups[(pq.source_domain.value, pq.target_domain.value)].append((pq, src))

    tasks: List[AuditTask] = []

    for (source_agent, target_agent), group in groups.items():

        if len(group) == 1:
            pq, src = group[0]
            tasks.append(peer_query_to_audit_task(pq, src))
            continue

        # Multiple queries for the same pair — merge into one debate task
        lead_pq = group[0][0]   # most urgent after pre-sort

        combined_query = "\n".join(
            f"{i}. {pq.question}"
            for i, (pq, _) in enumerate(group, start=1)
        )

        # Merge evidence; deduplicate by 80-char fingerprint
        seen_fps: set[str] = set()
        combined_evidence: List[str] = []
        for pq, src in group:
            for chunk in _build_evidence_chunks(pq, src):
                fp = chunk[:80]
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    combined_evidence.append(chunk)

        tasks.append(make_audit_task(
            task_id=f"pq_{lead_pq.query_id}_m{len(group)}",
            initiating_agent=source_agent,
            target_agent=target_agent,
            query=combined_query,
            evidence_chunks=combined_evidence,
            domain_tags=[source_agent, target_agent],
        ))

        logger.debug(
            "[Bridge] Merged %d %s→%s queries → task pq_%s_m%d.",
            len(group), source_agent, target_agent,
            lead_pq.query_id, len(group),
        )

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def peer_query_to_audit_task(
    peer_query: PeerQuery,
    source_result: Optional[AgentResult] = None,
) -> AuditTask:
    """
    Convert a single Module B PeerQuery into a Module D AuditTask.

    Args:
        peer_query:    Cross-domain question produced by a domain agent.
        source_result: The AgentResult that emitted this PeerQuery. Providing
                       it enables evidence enrichment beyond the 500-char
                       snippet stored on PeerQuery itself.

    Returns:
        AuditTask ready for make_initial_global_state(audit_tasks=[...]).
    """
    evidence = _build_evidence_chunks(peer_query, source_result)

    if not evidence:
        logger.warning(
            "[Bridge] PeerQuery %s (%s→%s) has no evidence — "
            "mesh debate will have minimal context.",
            peer_query.query_id,
            peer_query.source_domain.value,
            peer_query.target_domain.value,
        )

    return make_audit_task(
        task_id=f"pq_{peer_query.query_id}",
        initiating_agent=peer_query.source_domain.value,
        target_agent=peer_query.target_domain.value,
        query=peer_query.question,
        evidence_chunks=evidence,
        domain_tags=[
            peer_query.source_domain.value,
            peer_query.target_domain.value,
        ],
    )


def agent_results_to_audit_tasks(
    results: Dict[DomainType, AgentResult],
    prioritise_critical: bool = True,
    deduplicate_pairs: bool = True,
) -> List[AuditTask]:
    """
    Convert all Module B AgentResults into a prioritised list of AuditTasks
    for Module D to dispatch to the Module C mesh.

    Args:
        results:             Output of module_b.agent_factory.run_all_agents().
        prioritise_critical: Sort tasks so CRITICAL / HIGH urgency queries run
                             first. Recommended: True. If a kill-switch fires
                             mid-run and truncates the queue, the highest-risk
                             debates will have already run.
        deduplicate_pairs:   Merge PeerQuery objects for the same (source,
                             target) pair into one AuditTask. Recommended: True.
                             Multiple concurrent unresolved threads between the
                             same pair trigger Module C's fragmentation
                             kill-switch immediately.

    Returns:
        List[AuditTask] sorted by urgency descending (when prioritise_critical
        is True), ready for make_initial_global_state(audit_tasks=...).
        Returns [] if no domain agent flagged any peer queries — meaning all
        risks were self-contained and require no cross-domain debate.
    """
    all_queries: List[tuple[PeerQuery, AgentResult]] = [
        (pq, result)
        for result in results.values()
        for pq in result.peer_queries
    ]

    if not all_queries:
        logger.info("[Bridge] No peer queries found — returning empty task list.")
        return []

    n = len(all_queries)
    logger.info(
        "[Bridge] %d peer quer%s from %d domain agents.",
        n, "y" if n == 1 else "ies", len(results),
    )

    # Sort by urgency BEFORE dedup/merge so that:
    #   (a) the overall task list is urgency-ordered, and
    #   (b) the "lead" query in each merged group is the most urgent one.
    if prioritise_critical:
        all_queries.sort(
            key=lambda x: _URGENCY_WEIGHT.get(x[0].urgency, 0),
            reverse=True,
        )

    tasks = (
        _merge_by_pair(all_queries)
        if deduplicate_pairs
        else [peer_query_to_audit_task(pq, src) for pq, src in all_queries]
    )

    logger.info("[Bridge] Produced %d AuditTask(s) for Module D.", len(tasks))
    return tasks
