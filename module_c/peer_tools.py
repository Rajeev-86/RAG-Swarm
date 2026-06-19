"""
module_c/peer_tools.py
──────────────────────
Tool factories for the P2P mesh communication protocol.

Each domain agent is equipped with:
  • send_peer_query  — forward a claim + evidence to a peer agent
  • resolve_debate   — close a debate thread with an agreed finding

Tools are bound-per-agent via factory functions so the LLM's tool-calling
trace clearly attributes which agent sent which message.

Usage (called by build_agent_node in agent_nodes.py):
    send_tool    = create_send_peer_query_tool("financial")
    resolve_tool = create_resolve_debate_tool("financial")
    bound_llm    = llm.bind_tools([send_tool, resolve_tool])
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.tools import StructuredTool

VALID_AGENTS = frozenset({"financial", "legal", "hr", "cybersecurity"})


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: send_peer_query
# ─────────────────────────────────────────────────────────────────────────────

def create_send_peer_query_tool(source_agent: str) -> StructuredTool:
    """
    Return a LangChain tool bound to *source_agent* that lets it send a
    query + supporting RAG evidence to a peer domain agent.

    The returned PeerMessage dict is merged into the target's inbox by the
    agent_node executor (see agent_nodes.py).
    """

    # 1. Define the raw python function WITHOUT the @tool decorator
    def send_peer_query(
        target_agent: str,
        query: str,
        evidence_chunks: List[str],
        debate_id: Optional[str] = None,
        turn_number: int = 1,
    ) -> dict:
        """
        Send a query with supporting evidence to a peer domain agent.

        Args:
            target_agent:    Recipient — one of: financial, legal, hr, cybersecurity.
            query:           The specific claim or cross-domain question.
            evidence_chunks: RAG-retrieved document chunks backing the query.
            debate_id:       Existing debate thread to continue (None = new).
            turn_number:     Current turn within this debate thread.

        Returns:
            PeerMessage dict routed into the target's inbox.
        """
        if target_agent not in VALID_AGENTS:
            raise ValueError(
                f"Unknown agent '{target_agent}'. Valid: {sorted(VALID_AGENTS)}"
            )
        if target_agent == source_agent:
            raise ValueError(
                f"Agent '{source_agent}' cannot send a peer query to itself."
            )

        return {
            "message_id": str(uuid.uuid4()),
            "debate_id": debate_id or str(uuid.uuid4()),
            "source_agent": source_agent,
            "target_agent": target_agent,
            "query": query,
            "evidence_chunks": evidence_chunks,
            "turn_number": turn_number,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # 2. Build the tool dynamically to guarantee the JSON schema binds the correct name
    return StructuredTool.from_function(
        func=send_peer_query,
        name=f"send_peer_query_from_{source_agent}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: resolve_debate
# ─────────────────────────────────────────────────────────────────────────────

def create_resolve_debate_tool(source_agent: str) -> StructuredTool:
    """
    Return a LangChain tool that lets *source_agent* close a debate thread
    by emitting an agreed-upon finding summary.

    The returned ResolvedFinding dict is appended to resolved_findings and
    the matching DebateThread is marked is_resolved=True by the agent_node.
    """

    # 1. Define the raw python function WITHOUT the @tool decorator
    def resolve_debate(debate_id: str, resolution_summary: str) -> dict:
        """
        Mark a debate thread as resolved with a concise finding summary.

        Args:
            debate_id:           ID of the thread to close.
            resolution_summary:  Concise statement of the agreed finding,
                                 citing the relevant clause/document.

        Returns:
            ResolvedFinding dict appended to MeshState.resolved_findings.
        """
        return {
            "debate_id": debate_id,
            "resolved_by": source_agent,
            "resolution_summary": resolution_summary,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # 2. Build the tool dynamically to guarantee the JSON schema binds the correct name
    return StructuredTool.from_function(
        func=resolve_debate,
        name=f"resolve_debate_from_{source_agent}"
    )