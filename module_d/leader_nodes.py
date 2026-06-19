"""
module_d/leader_nodes.py
─────────────────────────
All LangGraph node functions for the Leader Agent.

Node inventory
──────────────
  build_dispatch_mesh_node()  — factory → invokes Module C mesh sub-graph
  handle_mesh_result_node()   — inspects mesh exit state, detects kill-switch
  build_force_summary_node()  — factory → LLM-synthesised forced finding
  build_checkpoint_node()     — factory → JSON checkpoint written to disk
  reset_context_node()        — clears last_mesh_state (context window reset)
  advance_task_node()         — marks task done/force_resolved, bumps index
  check_complete_node()       — sets audit_complete when queue is exhausted

Factory pattern
───────────────
Nodes that require an LLM or external dependency are returned by a factory
function (build_*) so the caller can inject a mock for testing without
touching Ollama or Module C at all:

    node = build_dispatch_mesh_node(
        mesh_graph=mock_graph,
        _mesh_state_factory=lambda i, t, q, e: {...}
    )
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .checkpoint_manager import write_json_checkpoint
from .leader_state import (
    AuditTask,
    GlobalState,
    make_log_entry,
)

logger = logging.getLogger(__name__)

_FORCE_SUMMARY_PROMPT = """You are a senior M&A analyst summarising a PARTIALLY RESOLVED due diligence debate.
The debate was forcibly interrupted by the Leader Agent's kill switch before the agents reached consensus.

Your task: produce a concise, accurate forced summary.

STRICT RULES:
  - Only include claims supported by the provided partial findings.
  - Mark every unresolved point as "⚠ UNRESOLVED — requires follow-up investigation."
  - Do NOT fabricate clause references, monetary figures, or risk ratings.
  - Format: 3–6 bullet points; each ≤ 35 words.
  - Begin with "FORCED SUMMARY (Leader Agent):" on its own line."""


# ─────────────────────────────────────────────────────────────────────────────
# Node: dispatch_mesh
# ─────────────────────────────────────────────────────────────────────────────

def build_dispatch_mesh_node(
    mesh_graph=None,
    llm: Optional[BaseChatModel] = None,
    _mesh_state_factory: Optional[Callable] = None,
) -> Callable[[GlobalState], Dict[str, Any]]:
    """
    Factory: returns a LangGraph node that dispatches the current AuditTask
    to the Module C mesh sub-graph and captures the full MeshState result.
    """
    def dispatch_mesh_node(state: GlobalState) -> Dict[str, Any]:
        tasks: List[AuditTask] = state["audit_tasks"]
        idx: int = state["current_task_idx"]

        # Guard: no more tasks → signal completion immediately.
        if idx >= len(tasks):
            return {"audit_complete": True}

        task = tasks[idx]

        # ── resolve dependencies (lazy import avoids hard coupling at import) ─
        graph = mesh_graph
        if graph is None:
            from module_c import build_mesh_graph  # type: ignore
            graph = build_mesh_graph(llm=llm)

        factory = _mesh_state_factory
        if factory is None:
            from module_c import create_initial_mesh_state  # type: ignore
            factory = create_initial_mesh_state

        # ── build seed state and invoke mesh ──────────────────────────────────
        mesh_seed = factory(
            task["initiating_agent"],
            task["target_agent"],
            task["query"],
            task["evidence_chunks"],
        )

        run_cfg = {
            "configurable": {
                "thread_id": f"{state['run_id']}:task:{task['task_id']}"
            }
        }
        mesh_result = graph.invoke(mesh_seed, config=run_cfg)

        # Mark task as running (will be updated to complete/force_resolved later)
        updated_tasks = list(tasks)
        updated_tasks[idx] = {**task, "status": "running"}

        return {
            "audit_tasks": updated_tasks,
            "last_mesh_state": dict(mesh_result),
            "audit_log": [
                make_log_entry(
                    "MESH_DISPATCHED",
                    (
                        f"Task '{task['task_id']}': "
                        f"{task['initiating_agent']} → {task['target_agent']}. "
                        f"Mesh ticks: {mesh_result.get('mesh_tick', 'n/a')}."
                    ),
                    task_id=task["task_id"],
                )
            ],
        }

    dispatch_mesh_node.__name__ = "dispatch_mesh_node"
    return dispatch_mesh_node


# ─────────────────────────────────────────────────────────────────────────────
# Node: handle_mesh_result
# ─────────────────────────────────────────────────────────────────────────────

def handle_mesh_result_node(state: GlobalState) -> Dict[str, Any]:
    """
    Inspect the mesh sub-graph exit state and branch accordingly.
    """
    mesh: Dict[str, Any] = state.get("last_mesh_state") or {}
    tasks = state["audit_tasks"]
    idx = state["current_task_idx"]
    task = tasks[idx] if idx < len(tasks) else None
    task_id = task["task_id"] if task else None

    kill_switch: bool = mesh.get("kill_switch_triggered", False)
    reason: str = mesh.get("kill_switch_reason") or ""
    findings: List[Dict] = mesh.get("resolved_findings", [])

    if kill_switch:
        # Don't absorb findings yet — force_summary will do a proper synthesis.
        return {
            "audit_log": [
                make_log_entry(
                    "KILL_SWITCH_FIRED",
                    f"Reason: {reason}  |  Partial findings available: {len(findings)}.",
                    task_id=task_id,
                    debate_id=mesh.get("kill_switch_debate_id"),
                )
            ],
        }

    # ── Clean exit: absorb findings into SharedMemory ─────────────────────────
    updated_memory = dict(state.get("shared_memory", {}))
    for finding in findings:
        updated_memory[finding["debate_id"]] = finding

    return {
        "shared_memory": updated_memory,
        "audit_log": [
            make_log_entry(
                "MESH_COMPLETE",
                f"Task '{task_id}': {len(findings)} finding(s) cleanly resolved.",
                task_id=task_id,
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: force_summary
# ─────────────────────────────────────────────────────────────────────────────

def build_force_summary_node(
    llm: Optional[BaseChatModel] = None,
) -> Callable[[GlobalState], Dict[str, Any]]:
    """
    Factory: returns a node that uses an LLM to synthesise a forced finding
    summary from whatever partial findings the interrupted mesh produced.
    """
    if llm is None:
        llm_backend = os.getenv("LLM_BACKEND", "groq").lower()
        if llm_backend == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192") 
            )
        elif llm_backend == "ollama":
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="llama3")
        else:
            raise ValueError(f"Unsupported LLM_BACKEND: {llm_backend}")

    def force_summary_node(state: GlobalState) -> Dict[str, Any]:
        mesh: Dict[str, Any] = state.get("last_mesh_state") or {}
        tasks = state["audit_tasks"]
        idx = state["current_task_idx"]
        task = tasks[idx] if idx < len(tasks) else None
        task_id = task["task_id"] if task else "unknown"

        kill_reason = mesh.get("kill_switch_reason") or "Unknown kill-switch reason."
        partial_findings: List[Dict] = mesh.get("resolved_findings", [])
        debate_id = mesh.get("kill_switch_debate_id") or f"forced_{task_id}"

        human_content = (
            f"KILL-SWITCH REASON:\n{kill_reason}\n\n"
            f"PARTIAL FINDINGS ({len(partial_findings)}):\n"
            + (
                "\n".join(
                    f"  [{i + 1}] (resolved by {f.get('resolved_by', '?')}) "
                    f"{f.get('resolution_summary', '')}"
                    for i, f in enumerate(partial_findings)
                )
                or "  (none — debate was cut off before any finding was agreed)"
            )
        )

        response = llm.invoke(
            [
                SystemMessage(content=_FORCE_SUMMARY_PROMPT),
                HumanMessage(content=human_content),
            ]
        )
        forced_text: str = getattr(response, "content", str(response))

        forced_finding: Dict[str, Any] = {
            "debate_id": debate_id,
            "resolved_by": "leader",
            "resolution_summary": forced_text,
            "timestamp": datetime.utcnow().isoformat(),
            "is_forced": True,
            "partial_findings_count": len(partial_findings),
        }

        # Absorb into SharedMemory
        updated_memory = dict(state.get("shared_memory", {}))
        updated_memory[debate_id] = forced_finding

        return {
            "shared_memory": updated_memory,
            "audit_log": [
                make_log_entry(
                    "FORCE_SUMMARY_GENERATED",
                    (
                        f"Leader forced summary for task '{task_id}' "
                        f"(debate_id={debate_id}). "
                        f"Summary length: {len(forced_text)} chars."
                    ),
                    task_id=task_id,
                    debate_id=debate_id,
                )
            ],
        }

    force_summary_node.__name__ = "force_summary_node"
    return force_summary_node


# ─────────────────────────────────────────────────────────────────────────────
# Node: checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpoint_node(
    checkpoint_dir: str = "checkpoints",
) -> Callable[[GlobalState], Dict[str, Any]]:
    """
    Factory: returns a node that writes a JSON checkpoint to disk and
    increments the checkpoint counter.
    """
    def checkpoint_node(state: GlobalState) -> Dict[str, Any]:
        mesh = state.get("last_mesh_state") or {}
        reason = mesh.get("kill_switch_reason") or "kill_switch"

        filepath = write_json_checkpoint(
            state=dict(state),
            checkpoint_dir=checkpoint_dir,
            reason=reason,
        )

        new_count = state.get("checkpoint_count", 0) + 1

        return {
            "checkpoint_count": new_count,
            "last_checkpoint_path": filepath,
            "audit_log": [
                make_log_entry(
                    "CHECKPOINT_SAVED",
                    f"Checkpoint #{new_count} → {filepath}",
                )
            ],
        }

    checkpoint_node.__name__ = "checkpoint_node"
    return checkpoint_node


# ─────────────────────────────────────────────────────────────────────────────
# Node: reset_context
# ─────────────────────────────────────────────────────────────────────────────

def reset_context_node(state: GlobalState) -> Dict[str, Any]:
    """
    Clear last_mesh_state, wiping all sub-agent context windows.
    """
    return {
        "last_mesh_state": None,
        "audit_log": [
            make_log_entry(
                "CONTEXT_WINDOWS_RESET",
                "last_mesh_state cleared. Sub-agent context windows wiped.",
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: advance_task
# ─────────────────────────────────────────────────────────────────────────────

def advance_task_node(state: GlobalState) -> Dict[str, Any]:
    """
    Mark the current task complete (clean) or force_resolved (kill-switch),
    advance current_task_idx, and update data_room_status counters.
    """
    tasks = list(state["audit_tasks"])
    idx = state["current_task_idx"]
    mesh = state.get("last_mesh_state") or {}
    kill_switch: bool = mesh.get("kill_switch_triggered", False)

    if idx < len(tasks):
        new_status = "force_resolved" if kill_switch else "complete"
        tasks[idx] = {**tasks[idx], "status": new_status}

    # ── update data_room_status counters ─────────────────────────────────────
    drs = {k: dict(v) for k, v in (state.get("data_room_status") or {}).items()}
    if idx < len(tasks):
        for domain in tasks[idx].get("domain_tags", []):
            if domain in drs:
                drs[domain]["tasks_complete"] += 1
                if kill_switch:
                    drs[domain]["tasks_force_resolved"] += 1
                    drs[domain]["has_escalations"] = True

    task_id = tasks[idx]["task_id"] if idx < len(tasks) else "n/a"

    return {
        "audit_tasks": tasks,
        "current_task_idx": idx + 1,
        "data_room_status": drs,
        "audit_log": [
            make_log_entry(
                "TASK_ADVANCED",
                (
                    f"Task '{task_id}' → '{tasks[idx]['status']}'. "
                    f"Next idx: {idx + 1}/{len(tasks)}."
                ),
                task_id=task_id,
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: check_complete
# ─────────────────────────────────────────────────────────────────────────────

def check_complete_node(state: GlobalState) -> Dict[str, Any]:
    """
    Set audit_complete=True when all tasks have been processed.
    Appends the final AUDIT_COMPLETE log entry when done.
    """
    idx = state["current_task_idx"]
    total = len(state.get("audit_tasks") or [])
    complete = idx >= total

    updates: Dict[str, Any] = {"audit_complete": complete}

    if complete:
        force_resolved = sum(
            1 for t in (state.get("audit_tasks") or [])
            if t.get("status") == "force_resolved"
        )
        updates["audit_log"] = [
            make_log_entry(
                "AUDIT_COMPLETE",
                (
                    f"All {total} task(s) processed. "
                    f"Clean: {total - force_resolved}  |  "
                    f"Force-resolved: {force_resolved}  |  "
                    f"Findings in shared_memory: {len(state.get('shared_memory') or {})}."
                ),
            )
        ]

    return updates