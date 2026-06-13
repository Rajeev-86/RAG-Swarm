"""
module_d — Leader Agent (State Hub)
=====================================
Public surface for the application layer and integration tests.

Quick-start
───────────
    from module_d import build_leader_graph, make_initial_global_state, make_audit_task

    tasks = [
        make_audit_task(
            task_id="t1",
            initiating_agent="financial",
            target_agent="legal",
            query="Do penalty clauses cover the $12M revenue gap?",
            evidence_chunks=["§4.2 APA: penalty threshold 5%"],
        ),
    ]

    graph  = build_leader_graph()
    state  = make_initial_global_state(tasks, run_id="audit-run-001")
    result = graph.invoke(state, config={"configurable": {"thread_id": state["run_id"]}})

    # Results
    result["shared_memory"]         # all cross-domain findings
    result["audit_log"]             # immutable event history
    result["data_room_status"]      # per-domain completion statistics
    result["last_checkpoint_path"]  # latest JSON checkpoint path (if any)
    result["checkpoint_count"]      # how many kill-switch events occurred

Resuming after interruption
────────────────────────────
    from module_d import load_latest_checkpoint

    ckpt = load_latest_checkpoint(run_id="audit-run-001")
    # ckpt["audit_log"], ckpt["partial_findings"], ckpt["kill_switch_reason"]
"""

from .checkpoint_manager import load_latest_checkpoint
from .leader_graph import build_leader_graph
from .leader_state import (
    AuditLogEntry,
    AuditTask,
    DataRoomDomainStatus,
    EVENT_TYPES,
    GlobalState,
    make_audit_task,
    make_initial_global_state,
    make_log_entry,
)

__all__ = [
    # Graph builder
    "build_leader_graph",
    # State types
    "GlobalState",
    "AuditTask",
    "AuditLogEntry",
    "DataRoomDomainStatus",
    "EVENT_TYPES",
    # Factory helpers
    "make_initial_global_state",
    "make_audit_task",
    "make_log_entry",
    # Checkpoint inspection
    "load_latest_checkpoint",
]
