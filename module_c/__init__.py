"""
module_c — Mesh Interface (P2P Debate Layer)
============================================
Public surface exposed to Module D (Leader Agent) and tests.

Quick-start
───────────
    from module_c import build_mesh_graph, create_initial_mesh_state

    mesh  = build_mesh_graph()               # Ollama llama3:8b by default
    state = create_initial_mesh_state(
        initiating_agent="financial",
        target_agent="legal",
        query="Do penalty clauses cover the $12 M revenue gap?",
        evidence_chunks=["§4.2 APA: penalty threshold 5 %"],
    )
    result = mesh.invoke(state, config={"configurable": {"thread_id": "run-001"}})

    # Module D reads:
    result["kill_switch_triggered"]   # bool
    result["kill_switch_reason"]      # str | None
    result["resolved_findings"]       # List[ResolvedFinding]
"""

from .debate_monitor import MAX_TURNS
from .mesh_graph import build_mesh_graph, create_initial_mesh_state
from .mesh_state import DebateThread, MeshState, PeerMessage, ResolvedFinding
from .peer_tools import create_resolve_debate_tool, create_send_peer_query_tool

__all__ = [
    # Graph builder + state bootstrapper
    "build_mesh_graph",
    "create_initial_mesh_state",
    # State types
    "MeshState",
    "PeerMessage",
    "DebateThread",
    "ResolvedFinding",
    # Tool factories (useful if Module D needs to extend them)
    "create_send_peer_query_tool",
    "create_resolve_debate_tool",
    # Constants
    "MAX_TURNS",
]
