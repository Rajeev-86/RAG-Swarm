"""
module_d/leader_state.py
─────────────────────────
GlobalState TypedDict for the Leader Agent (Module D).

Spec §3.2 mandates three core state objects carried by the Leader:
  • SharedMemory       — cross-domain findings accumulated across all debates
  • Data_Room_Status   — per-domain audit progress
  • Audit_Log          — append-only event trace (LangGraph reducer: operator.add)

The remaining fields are operational metadata for the task queue, mesh I/O,
and checkpoint bookkeeping.
"""
from __future__ import annotations

import operator
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Sub-types
# ─────────────────────────────────────────────────────────────────────────────

class AuditTask(TypedDict):
    """
    One cross-domain debate task dispatched to the Mesh sub-graph.
    Created by the application layer (or a future planning node).
    """
    task_id: str
    initiating_agent: str    # "financial" | "legal" | "hr" | "cyber"
    target_agent: str
    query: str
    evidence_chunks: List[str]
    status: str              # "pending" | "running" | "complete" | "force_resolved"
    domain_tags: List[str]   # domains touched, e.g. ["financial", "legal"]


class AuditLogEntry(TypedDict):
    """
    Immutable event record appended to audit_log. Never edited or removed.
    The log is the authoritative trace of every Leader decision.
    """
    timestamp: str           # UTC ISO-8601
    event_type: str          # one of EVENT_TYPES below
    task_id: Optional[str]
    debate_id: Optional[str]
    detail: str


class DataRoomDomainStatus(TypedDict):
    """Per-domain audit progress counter (the Data_Room_Status object in spec §3.2)."""
    domain: str
    tasks_total: int
    tasks_complete: int
    tasks_force_resolved: int
    has_escalations: bool


# Canonical event_type strings — used in AuditLogEntry and asserted in tests.
EVENT_TYPES = frozenset({
    "AUDIT_STARTED",
    "MESH_DISPATCHED",
    "MESH_COMPLETE",
    "KILL_SWITCH_FIRED",
    "FORCE_SUMMARY_GENERATED",
    "CHECKPOINT_SAVED",
    "CONTEXT_WINDOWS_RESET",
    "TASK_ADVANCED",
    "AUDIT_COMPLETE",
})


# ─────────────────────────────────────────────────────────────────────────────
# Root state (the LangGraph StateGraph schema)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalState(TypedDict):
    """
    Root state for the Leader Agent's LangGraph StateGraph.

    On audit completion the application layer reads:
        result["shared_memory"]       — all cross-domain findings keyed by debate_id
        result["audit_log"]           — full immutable event history
        result["data_room_status"]    — per-domain completion statistics
        result["last_checkpoint_path"] — path to last JSON checkpoint (if any)
    """

    # ── §3.2 core state objects ───────────────────────────────────────────────
    shared_memory: Dict[str, Any]                        # keyed by debate_id
    data_room_status: Dict[str, DataRoomDomainStatus]    # keyed by domain name
    audit_log: Annotated[List[AuditLogEntry], operator.add]   # append-only

    # ── task queue ───────────────────────────────────────────────────────────
    audit_tasks: List[AuditTask]
    current_task_idx: int

    # ── mesh sub-graph I/O ───────────────────────────────────────────────────
    # Full MeshState dict returned by the last Module C invocation.
    # The Leader reads kill_switch_triggered, kill_switch_reason,
    # kill_switch_debate_id, and resolved_findings from here.
    # Cleared to None by reset_context_node after each kill-switch event.
    last_mesh_state: Optional[Dict[str, Any]]

    # ── checkpoint metadata ──────────────────────────────────────────────────
    checkpoint_count: int
    last_checkpoint_path: Optional[str]

    # ── lifecycle ────────────────────────────────────────────────────────────
    run_id: str
    audit_complete: bool


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers (used by nodes, tests, and the application layer)
# ─────────────────────────────────────────────────────────────────────────────

def make_log_entry(
    event_type: str,
    detail: str,
    task_id: Optional[str] = None,
    debate_id: Optional[str] = None,
) -> AuditLogEntry:
    """Convenience constructor for a single AuditLogEntry."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "task_id": task_id,
        "debate_id": debate_id,
        "detail": detail,
    }


def make_audit_task(
    task_id: str,
    initiating_agent: str,
    target_agent: str,
    query: str = "Placeholder cross-domain query.",
    evidence_chunks: Optional[List[str]] = None,
    domain_tags: Optional[List[str]] = None,
) -> AuditTask:
    """Convenience constructor for an AuditTask."""
    return {
        "task_id": task_id,
        "initiating_agent": initiating_agent,
        "target_agent": target_agent,
        "query": query,
        "evidence_chunks": evidence_chunks or [],
        "status": "pending",
        "domain_tags": domain_tags or sorted({initiating_agent, target_agent}),
    }


def make_initial_global_state(
    audit_tasks: List[AuditTask],
    run_id: Optional[str] = None,
) -> GlobalState:
    """
    Bootstrap a fresh GlobalState for a new audit run.
    Called by the application layer before the first graph.invoke().

    Args:
        audit_tasks:  Pre-planned list of cross-domain debate tasks.
                      Build with make_audit_task(), or feed from Module A's
                      RAG pipeline output.
        run_id:       UUID string identifying this run. Auto-generated if None.

    Returns:
        A fully-initialised GlobalState dict ready for graph.invoke().
    """
    run_id = run_id or str(uuid.uuid4())

    # Derive data_room_status from the task domain_tags
    domain_counts: Dict[str, int] = {}
    for task in audit_tasks:
        for domain in task.get("domain_tags", []):
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    data_room_status: Dict[str, DataRoomDomainStatus] = {
        domain: {
            "domain": domain,
            "tasks_total": count,
            "tasks_complete": 0,
            "tasks_force_resolved": 0,
            "has_escalations": False,
        }
        for domain, count in domain_counts.items()
    }

    return {
        "shared_memory": {},
        "data_room_status": data_room_status,
        "audit_log": [
            make_log_entry(
                "AUDIT_STARTED",
                f"run_id={run_id}. {len(audit_tasks)} task(s) queued.",
            )
        ],
        "audit_tasks": list(audit_tasks),
        "current_task_idx": 0,
        "last_mesh_state": None,
        "checkpoint_count": 0,
        "last_checkpoint_path": None,
        "run_id": run_id,
        "audit_complete": False,
    }
