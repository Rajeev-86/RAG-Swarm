"""
module_d/leader_graph.py
─────────────────────────
Assembles the Leader Agent's LangGraph StateGraph (Module D).

Full topology
─────────────

    START
      │
      ▼
  dispatch_mesh ──────────────────────────────────────────────────────────┐
      │                                                               (loop)
      ▼
  handle_mesh_result
      │
      ├──[kill_switch=True]──► force_summary ──► checkpoint ──► advance_task
      │                                                               │
      └──[kill_switch=False]──────────────────────────────────► advance_task
                                                                      │
                                                          ┌───────────┴───────────┐
                                                     [ks=True]               [ks=False]
                                                          │                       │
                                                    reset_context           check_complete
                                                          │                       │
                                                    check_complete          ┌─────┴──────┐
                                                          │                loop         end
                                                    ┌─────┴──────┐
                                                  loop           end

Key design decisions
─────────────────────
• advance_task runs on BOTH paths so DRS counters and task status are always
  updated before the kill-switch flag is cleared.

• reset_context runs AFTER advance_task so advance_task can still read
  kill_switch_triggered from last_mesh_state.

• check_complete is reachable from both advance_task (clean path) and
  reset_context (kill-switch path) — same conditional edge handles both.

Integration
───────────
    from module_d import build_leader_graph, make_initial_global_state, make_audit_task

    tasks = [
        make_audit_task("t1", "financial", "legal",
                        query="Do penalty clauses cover the $12M gap?",
                        evidence_chunks=rag_chunks),
    ]
    graph  = build_leader_graph()
    state  = make_initial_global_state(tasks, run_id="audit-run-001")
    result = graph.invoke(state, config={"configurable": {"thread_id": state["run_id"]}})

    # Outputs
    result["shared_memory"]        # all findings, keyed by debate_id
    result["audit_log"]            # full event history
    result["last_checkpoint_path"] # path to last JSON checkpoint (if any)
"""
from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .checkpoint_manager import get_checkpointer
from .leader_nodes import (
    advance_task_node,
    build_checkpoint_node,
    build_dispatch_mesh_node,
    build_force_summary_node,
    check_complete_node,
    handle_mesh_result_node,
    reset_context_node,
)
from .leader_state import GlobalState


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions (conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_mesh_result(
    state: GlobalState,
) -> Literal["force_summary", "advance_task"]:
    """
    After handle_mesh_result: branch on kill_switch_triggered.

    kill_switch=True  → force_summary (synthesise finding + checkpoint path)
    kill_switch=False → advance_task  (clean exit path)
    """
    mesh = state.get("last_mesh_state") or {}
    return "force_summary" if mesh.get("kill_switch_triggered") else "advance_task"


def _route_after_advance_task(
    state: GlobalState,
) -> Literal["reset_context", "check_complete"]:
    """
    After advance_task: decide whether context needs wiping.

    kill_switch was True  → reset_context (wipe mesh state before next dispatch)
    kill_switch was False → check_complete (no wipe needed)

    NOTE: last_mesh_state is still set at this point; reset_context clears it
    afterward, which is why advance_task must precede reset_context.
    """
    mesh = state.get("last_mesh_state") or {}
    return "reset_context" if mesh.get("kill_switch_triggered") else "check_complete"


def _route_after_check_complete(
    state: GlobalState,
) -> Literal["dispatch_mesh", "__end__"]:
    """
    After check_complete: loop if tasks remain, otherwise end the audit.
    """
    return "__end__" if state.get("audit_complete") else "dispatch_mesh"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_leader_graph(
    llm: Optional[BaseChatModel] = None,
    mesh_graph=None,
    checkpoint_dir: str = "checkpoints",
    use_sqlite_checkpointer: bool = True,
    _mesh_state_factory=None,  # injectable for testing; see build_dispatch_mesh_node
) -> Any:  # langgraph.graph.CompiledGraph
    """
    Assemble and compile the Leader Agent's StateGraph.

    Args:
        llm:                     Chat model for force_summary (and mesh agents if
                                 mesh_graph is None). Dynamically routes via LLM_BACKEND
                                 if None is provided.
        mesh_graph:              Pre-compiled Module C graph.  If None, built lazily
                                 from module_c.build_mesh_graph(llm=llm) on first call.
        checkpoint_dir:          Directory for JSON snapshots and SQLite db.
        use_sqlite_checkpointer: True  → SqliteSaver. False → MemorySaver.
        _mesh_state_factory:     For tests only.
    """
    builder = StateGraph(GlobalState)

    # ── Ensure global LLM instantiation if none provided ──────────────────────
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

    # ── instantiate node functions ────────────────────────────────────────────
    dispatch_mesh = build_dispatch_mesh_node(
        mesh_graph=mesh_graph,
        llm=llm,
        _mesh_state_factory=_mesh_state_factory,
    )
    # The forced summary node now explicitly receives the validated LLM
    force_summary = build_force_summary_node(llm=llm)
    checkpoint    = build_checkpoint_node(checkpoint_dir=checkpoint_dir)

    # ── add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("dispatch_mesh",      dispatch_mesh)
    builder.add_node("handle_mesh_result", handle_mesh_result_node)
    builder.add_node("force_summary",      force_summary)
    builder.add_node("checkpoint",         checkpoint)
    builder.add_node("advance_task",       advance_task_node)
    builder.add_node("reset_context",      reset_context_node)
    builder.add_node("check_complete",     check_complete_node)

    # ── edges ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("dispatch_mesh")
    builder.add_edge("dispatch_mesh", "handle_mesh_result")
    builder.add_conditional_edges(
        "handle_mesh_result",
        _route_after_mesh_result,
        {"force_summary": "force_summary", "advance_task": "advance_task"},
    )
    builder.add_edge("force_summary", "checkpoint")
    builder.add_edge("checkpoint",    "advance_task")
    builder.add_conditional_edges(
        "advance_task",
        _route_after_advance_task,
        {"reset_context": "reset_context", "check_complete": "check_complete"},
    )
    builder.add_edge("reset_context", "check_complete")
    builder.add_conditional_edges(
        "check_complete",
        _route_after_check_complete,
        {"dispatch_mesh": "dispatch_mesh", "__end__": END},
    )

    # ── checkpointer ──────────────────────────────────────────────────────────
    if use_sqlite_checkpointer:
        checkpointer = get_checkpointer(checkpoint_dir=checkpoint_dir)
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)