"""
module_c/mesh_state.py
──────────────────────
Typed state schema for the Mesh sub-graph (Module C).

Design notes
────────────
• Per-agent inboxes use plain List (replace semantics) so each node can
  atomically deliver one message and the old payload is discarded.
• `debate_threads` uses a custom merge reducer so a node updating a single
  thread doesn't accidentally wipe the others.
• `resolved_findings` uses operator.add (append) to accumulate findings
  across the full mesh run — the Leader Agent reads this on exit.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class PeerMessage(TypedDict):
    """A single P2P message exchanged between domain agents."""

    message_id: str
    debate_id: str
    source_agent: str           # "financial" | "legal" | "hr" | "cyber"
    target_agent: str
    query: str                  # the claim or cross-domain question
    evidence_chunks: List[str]  # RAG-retrieved chunks backing the query
    turn_number: int
    timestamp: str              # UTC ISO-8601


class DebateThread(TypedDict):
    """Lifecycle record for one P2P debate exchange."""

    debate_id: str
    initiating_agent: str
    responding_agent: str
    turn_count: int
    is_resolved: bool
    resolution_summary: Optional[str]
    is_escalated: bool          # True after Leader kill-switch fires


class ResolvedFinding(TypedDict):
    """An agreed-upon finding emitted by two agents (or forced by Leader)."""

    debate_id: str
    resolved_by: str            # agent name, or "leader" for forced summary
    resolution_summary: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Reducer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_debate_threads(
    old: Dict[str, DebateThread],
    new: Dict[str, DebateThread],
) -> Dict[str, DebateThread]:
    """
    Merge-update debate threads.

    Nodes return only the threads they modified; this reducer ensures a
    partial update doesn't overwrite unrelated threads.
    """
    return {**old, **new}


# ─────────────────────────────────────────────────────────────────────────────
# Mesh sub-graph state
# ─────────────────────────────────────────────────────────────────────────────

class MeshState(TypedDict):
    """
    Shared state for the Mesh sub-graph.

    Consumed by:
      • domain agent nodes (financial, legal, hr, cyber)
      • mesh_router node
      • debate_monitor node
      • Leader Agent (Module D) on EXIT — reads kill_switch_* and resolved_findings
    """

    # ── per-agent inboxes (plain list → replace semantics) ───────────────────
    financial_inbox: List[PeerMessage]
    legal_inbox: List[PeerMessage]
    hr_inbox: List[PeerMessage]
    cyber_inbox: List[PeerMessage]

    # ── debate bookkeeping (merge reducer) ───────────────────────────────────
    debate_threads: Annotated[Dict[str, DebateThread], _merge_debate_threads]

    # ── router metadata ──────────────────────────────────────────────────────
    active_agents: List[str]    # agents with pending mail this tick
    mesh_tick: int

    # ── kill-switch signals (read by Leader / Module D) ──────────────────────
    kill_switch_triggered: bool
    kill_switch_reason: Optional[str]
    kill_switch_debate_id: Optional[str]

    # ── output (append reducer) ──────────────────────────────────────────────
    resolved_findings: Annotated[List[ResolvedFinding], operator.add]
