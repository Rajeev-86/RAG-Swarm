"""
module_c/mesh_graph.py
──────────────────────
Assembles the Mesh sub-graph (Module C) as a compiled LangGraph StateGraph.

Topology
────────

    START
      │
      ▼
  mesh_router ──(conditional, picks first active agent)──► financial
      ▲                                                  │  legal
      │                                                  │  hr
      │                                                  │  cybersecurity
      │                                                  │
      │         ◄─────────────────────────────────────── ┘
      │                (each agent → debate_monitor)
      │
  debate_monitor
      │
      ├──► "continue_mesh"       → mesh_router   (loop)
      └──► "escalate_to_leader"  → END

Sequential single-agent-per-tick model
───────────────────────────────────────
The router activates the first agent with pending mail each tick.  This keeps
execution traces linear and easy to follow in LangSmith/Phoenix.

To upgrade to true parallelism later, replace _route_to_one_active_agent()
with a Send-based fan-out:

    from langgraph.types import Send
    def fan_out(state): return [Send(a, state) for a in state["active_agents"]]
    builder.add_conditional_edges("mesh_router", fan_out)

Module D (Leader Agent) integration
─────────────────────────────────────
    from module_c import build_mesh_graph, create_initial_mesh_state

    mesh = build_mesh_graph()
    seed  = create_initial_mesh_state(
        initiating_agent="financial",
        target_agent="legal",
        query="Do penalty clauses cover the $12 M revenue gap?",
        evidence_chunks=["§4.2 APA: penalty threshold 5 %"]
    )
    result = mesh.invoke(seed, config={"configurable": {"thread_id": "run-001"}})

    if result["kill_switch_triggered"]:
        # Leader forces a summary and checkpoints
        ...
    else:
        findings = result["resolved_findings"]
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agent_nodes import build_agent_node
from .debate_monitor import debate_monitor_node, should_continue_mesh
from .mesh_state import DebateThread, MeshState, PeerMessage

_DOMAIN_AGENTS = ("financial", "legal", "hr", "cybersecurity")


# ─────────────────────────────────────────────────────────────────────────────
# Internal orchestration nodes
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_router_node(state: MeshState) -> Dict[str, Any]:
    """
    Scan inboxes, publish the list of agents with pending mail, and
    advance the tick counter for trace readability.
    """
    active = [a for a in _DOMAIN_AGENTS if state.get(f"{a}_inbox")]
    return {
        "active_agents": active,
        "mesh_tick": state.get("mesh_tick", 0) + 1,
    }


def _route_to_one_active_agent(state: MeshState) -> str:
    """
    Return the name of the first agent with pending mail.
    Falls back to "debate_monitor" if no inboxes are populated so the
    monitor can trigger a clean exit rather than silently stalling.
    """
    active = state.get("active_agents", [])
    return active[0] if active else "debate_monitor"


# ─────────────────────────────────────────────────────────────────────────────
# Public graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh_graph(
    llm: Optional[BaseChatModel] = None,
    checkpointer=None,
) -> Any:  # langgraph.graph.CompiledGraph
    """
    Assemble and compile the Mesh sub-graph.

    Args:
        llm:          LangChain chat model for all domain agents.
                      Defaults to Ollama llama3:8b.
                      Swap to Groq for faster dev/test cycles:
                          from langchain_groq import ChatGroq
                          llm = ChatGroq(model="llama-3.1-8b-instant")
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to MemorySaver (in-process).
                      Pass a SqliteSaver or PostgresSaver for cross-session
                      durability (required by Module D's checkpoint spec).

    Returns:
        A compiled CompiledGraph ready for .invoke() / .stream().
    """
    builder = StateGraph(MeshState)

    # ── domain agent nodes ───────────────────────────────────────────────────
    for agent_name in _DOMAIN_AGENTS:
        builder.add_node(agent_name, build_agent_node(agent_name, llm=llm))

    # ── orchestration nodes ──────────────────────────────────────────────────
    builder.add_node("mesh_router", _mesh_router_node)
    builder.add_node("debate_monitor", debate_monitor_node)

    # ── edges ────────────────────────────────────────────────────────────────
    builder.set_entry_point("mesh_router")

    # Router fans out to whichever agent has mail (or straight to monitor).
    builder.add_conditional_edges(
        "mesh_router",
        _route_to_one_active_agent,
        {
            **{a: a for a in _DOMAIN_AGENTS},
            "debate_monitor": "debate_monitor",   # idle-inbox fallback
        },
    )

    # All domain agents report back to the monitor after each turn.
    for agent_name in _DOMAIN_AGENTS:
        builder.add_edge(agent_name, "debate_monitor")

    # Monitor decides: loop or hand off to Leader.
    builder.add_conditional_edges(
        "debate_monitor",
        should_continue_mesh,
        {
            "continue_mesh": "mesh_router",
            "escalate_to_leader": END,
        },
    )

    return builder.compile(checkpointer=checkpointer or MemorySaver())


# ─────────────────────────────────────────────────────────────────────────────
# Public helper: bootstrap state (called by Leader Agent / Module D)
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_mesh_state(
    initiating_agent: str,
    target_agent: str,
    query: str,
    evidence_chunks: List[str],
) -> MeshState:
    """
    Create a fresh MeshState seeded with a single opening P2P query.

    Called by the Leader Agent (Module D) when it delegates a cross-domain
    conflict to the mesh for resolution.

    Args:
        initiating_agent: Agent kicking off the debate (e.g., "financial").
        target_agent:     Agent receiving the first query (e.g., "legal").
        query:            The specific claim or cross-domain question.
        evidence_chunks:  RAG-retrieved context backing the opening query.

    Returns:
        A fully initialised MeshState dict ready for mesh_graph.invoke().

    Raises:
        ValueError: if initiating_agent == target_agent.
    """
    if initiating_agent == target_agent:
        raise ValueError(
            f"initiating_agent and target_agent must differ, got '{initiating_agent}' twice."
        )

    debate_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    seed_message: PeerMessage = {
        "message_id": str(uuid.uuid4()),
        "debate_id": debate_id,
        "source_agent": initiating_agent,
        "target_agent": target_agent,
        "query": query,
        "evidence_chunks": evidence_chunks,
        "turn_number": 1,
        "timestamp": now,
    }

    initial_thread: DebateThread = {
        "debate_id": debate_id,
        "initiating_agent": initiating_agent,
        "responding_agent": target_agent,
        "turn_count": 1,
        "is_resolved": False,
        "resolution_summary": None,
        "is_escalated": False,
    }

    state: MeshState = {
        "financial_inbox": [],
        "legal_inbox": [],
        "hr_inbox": [],
        "cybersecurity_inbox": [],
        "debate_threads": {debate_id: initial_thread},
        "active_agents": [target_agent],
        "mesh_tick": 0,
        "kill_switch_triggered": False,
        "kill_switch_reason": None,
        "kill_switch_debate_id": None,
        "resolved_findings": [],
    }

    # Seed the target's inbox with the opening query.
    state[f"{target_agent}_inbox"] = [seed_message]  # type: ignore[literal-required]
    return state
