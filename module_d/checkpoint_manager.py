"""
module_d/checkpoint_manager.py
────────────────────────────────
Disk persistence for the Leader Agent (spec §3.2: "saves the state to disk").

Two complementary mechanisms
──────────────────────────────
  1. LangGraph SqliteSaver  — binary LangGraph checkpoints; enables full
                              cross-session state recovery via .invoke() with
                              an existing thread_id.  Requires the optional
                              `langgraph-checkpoint-sqlite` package; falls back
                              to MemorySaver transparently in CI/test envs.

  2. JSON audit trail       — human-readable snapshot written alongside each
                              kill-switch event.  Carries partial findings,
                              kill-switch reason, and full audit_log so a human
                              analyst can inspect interrupted runs without
                              running Python at all.

Public API
──────────
    get_checkpointer()          → pass to build_leader_graph(checkpointer=...)
    write_json_checkpoint()     → called by checkpoint_node()
    load_latest_checkpoint()    → post-mortem inspection / audit resumption
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_CHECKPOINT_DIR = "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LangGraph checkpointer
# ─────────────────────────────────────────────────────────────────────────────

def get_checkpointer(
    checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
    db_filename: str = "leader_agent.db",
):
    """
    Return a LangGraph checkpointer for disk-persistent state recovery.

    Attempts SqliteSaver first (langgraph-checkpoint-sqlite).
    Silently falls back to MemorySaver when the package is absent — suitable
    for test environments and CI pipelines that don't need cross-session
    durability.

    The SQLite file is shared across all runs; individual runs are isolated
    by their thread_id (the run_id) inside LangGraph.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        db_path = str(Path(checkpoint_dir) / db_filename)
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    except (ImportError, AttributeError):
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Human-readable JSON audit trail
# ─────────────────────────────────────────────────────────────────────────────

def write_json_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
    reason: Optional[str] = None,
) -> str:
    """
    Write a JSON snapshot of GlobalState to disk.

    Called by checkpoint_node() immediately after each kill-switch event.

    File naming: checkpoint_{run_id[:8]}_{seq:04d}_{YYYYMMDDTHHMMSS}.json
    e.g.         checkpoint_a1b2c3d4_0001_20260101T120000.json

    Args:
        state:           The current GlobalState dict.
        checkpoint_dir:  Directory to write into (created if absent).
        reason:          Human-readable trigger description.  Defaults to the
                         kill_switch_reason from last_mesh_state.

    Returns:
        Absolute path of the written file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run_id = state.get("run_id", "unknown")
    seq = state.get("checkpoint_count", 0)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"checkpoint_{run_id[:8]}_{seq:04d}_{ts}.json"
    filepath = os.path.join(checkpoint_dir, filename)

    mesh = state.get("last_mesh_state") or {}
    snapshot = {
        "meta": {
            "run_id": run_id,
            "checkpoint_seq": seq,
            "timestamp": datetime.utcnow().isoformat(),
            "trigger": reason or mesh.get("kill_switch_reason") or "kill_switch",
        },
        "shared_memory": state.get("shared_memory", {}),
        "data_room_status": state.get("data_room_status", {}),
        "audit_log": state.get("audit_log", []),
        "current_task_idx": state.get("current_task_idx", 0),
        "kill_switch_reason": mesh.get("kill_switch_reason"),
        "kill_switch_debate_id": mesh.get("kill_switch_debate_id"),
        "partial_findings": mesh.get("resolved_findings", []),
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, default=str)

    return os.path.abspath(filepath)


def load_latest_checkpoint(
    checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
    run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the most recent JSON checkpoint, optionally filtered by run_id prefix.

    Useful for post-mortem inspection and for resuming interrupted audits.

    Returns:
        Parsed checkpoint dict, or None if no matching files exist.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        return None

    prefix = f"checkpoint_{run_id[:8]}_" if run_id else "checkpoint_"
    candidates = sorted(
        [f for f in path.glob("*.json") if f.name.startswith(prefix)],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        return None

    with open(candidates[0], "r", encoding="utf-8") as fh:
        return json.load(fh)
