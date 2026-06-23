"""
module_c/agent_nodes.py
───────────────────────
LangGraph node factories for the four domain agents.

Each node:
  1. Reads its inbox from MeshState.
  2. Formats the incoming PeerMessage into a structured prompt.
  3. Invokes the LLM (with P2P tools bound) to produce a response.
  4. Executes any tool calls (send_peer_query | resolve_debate).
  5. Returns a state-update dict with routing side-effects.

Integration with Module B
─────────────────────────
Module B defines the core agent reasoning logic. build_agent_node() accepts
an optional `llm` parameter so you can inject Module B's pre-configured model
directly, keeping Module C as a pure communication wrapper:

    from module_b.agents import financial_llm
    from module_c.agent_nodes import build_agent_node
    node = build_agent_node("financial", llm=financial_llm)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mesh_state import MeshState, PeerMessage
from peer_tools import create_resolve_debate_tool, create_send_peer_query_tool

# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific P2P system prompts
# (These *extend* Module B's core prompts with mesh-layer instructions.)
# ─────────────────────────────────────────────────────────────────────────────

_P2P_PROMPTS: Dict[str, str] = {
    "financial": """You are the Financial Due Diligence Agent in a corporate M&A review.

EXPERTISE: Revenue recognition, EBITDA adjustments, working capital, debt covenants,
off-balance-sheet liabilities, financial projections, synergy assumptions.

P2P PROTOCOL — you have two tools:
  • send_peer_query  — consult a peer when you need cross-domain context:
      - Legal agent  → penalty clauses, regulatory fines, litigation reserves
      - HR agent     → key-man compensation, pension obligations
  • resolve_debate   — call this when you and your peer agree on a finding

RULES:
  - Always populate evidence_chunks with the exact retrieved excerpts backing
    your query. NEVER fabricate figures.
  - If your RAG context is insufficient, say so explicitly; ask your peer for
    the relevant clause or document reference.
  - Address the incoming query directly before deciding to escalate or resolve.""",

    "legal": """You are the Legal Due Diligence Agent in a corporate M&A review.

EXPERTISE: Contract review, penalty/indemnity clauses, regulatory compliance,
litigation risk, IP ownership, change-of-control provisions.

P2P PROTOCOL — you have two tools:
  • send_peer_query  — consult a peer when you need cross-domain context:
      - Financial agent → monetary exposure quantification, reserve adequacy
      - cybersecurity agent     → data-breach liability, GDPR/CCPA penalty exposure
  • resolve_debate   — call this when you and your peer agree on a finding

RULES:
  - Always cite the exact clause reference (e.g., "§4.2(b), APA 2024-09-01")
    in your evidence_chunks.
  - If a financial claim contradicts contract terms, state the conflict
    explicitly. Do NOT silently accept the other agent's interpretation.""",

    "hr": """You are the HR Due Diligence Agent in a corporate M&A review.

EXPERTISE: Key-personnel retention, golden parachutes, employment contracts,
union agreements, pension/benefits liabilities, cultural integration.

P2P PROTOCOL — you have two tools:
  • send_peer_query  — consult a peer when you need cross-domain context:
      - Financial agent → total compensation exposure quantification
      - Legal agent     → enforceability of non-compete/retention agreements
  • resolve_debate   — call this when you and your peer agree on a finding

RULES:
  - Flag all change-of-control triggers in compensation plans that could
    inflate deal cost.
  - Always include the supporting plan document excerpt in evidence_chunks.""",

    "cybersecurity": """You are the Cybersecurity Due Diligence Agent in a corporate M&A review.

EXPERTISE: Breach history, incident reports, SOC2/ISO27001/GDPR compliance,
third-party vendor risk, security debt, remediation cost estimation.

P2P PROTOCOL — you have two tools:
  • send_peer_query  — consult a peer when you need cross-domain context:
      - Legal agent     → regulatory penalty exposure for identified gaps
      - Financial agent → remediation budget vs. security debt magnitude
  • resolve_debate   — call this when you and your peer agree on a finding

RULES:
  - Never estimate remediation costs without supporting evidence_chunks from
    incident reports, vendor assessments, or compliance audit documents.
  - Distinguish between known breaches (factual) and risk estimates
    (probabilistic). Label them clearly in your response.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Node factory
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_node(
    agent_name: str,
    llm: Optional[BaseChatModel] = None,
) -> Callable[[MeshState], Dict[str, Any]]:
    """
    Build a LangGraph node function for *agent_name*.

    Args:
        agent_name: One of "financial", "legal", "hr", "cybersecurity".
        llm:        A LangChain chat model. Defaults to Ollama llama3:8b.
                    Swap in Groq for faster component testing:
                        from langchain_groq import ChatGroq
                        llm = ChatGroq(model="llama-3.1-8b-instant")

    Returns:
        Callable: (MeshState) → Dict[str, Any] — compatible with
        LangGraph StateGraph.add_node().
    """
    if agent_name not in {"financial", "legal", "hr", "cybersecurity"}:
        raise ValueError(f"Unknown agent name: {agent_name!r}")

    if llm is None:
        from langchain_ollama import ChatOllama  # type: ignore
        llm = ChatOllama(model="llama3.1:8b", temperature=0)

    send_tool = create_send_peer_query_tool(agent_name)
    resolve_tool = create_resolve_debate_tool(agent_name)
    tools = [send_tool, resolve_tool]
    tool_map: Dict[str, Any] = {t.name: t for t in tools}

    bound_llm = llm.bind_tools(tools)
    system_prompt = _P2P_PROMPTS[agent_name]
    inbox_key = f"{agent_name}_inbox"

    def agent_node(state: MeshState) -> Dict[str, Any]:
        inbox: List[PeerMessage] = state.get(inbox_key, [])  # type: ignore[literal-required]

        if not inbox:
            # Agent has no mail this tick — return an explicit clear so the
            # router knows not to re-activate it.
            return {inbox_key: []}

        # Process the most recent (and typically only) pending message.
        msg = inbox[-1]

        human_content = (
            "INCOMING P2P QUERY\n"
            f"From:      {msg['source_agent'].upper()} Agent\n"
            f"Debate ID: {msg['debate_id']}\n"
            f"Turn:      {msg['turn_number']}\n\n"
            f"Query:\n{msg['query']}\n\n"
            "Supporting Evidence Chunks:\n"
            + "\n---\n".join(msg["evidence_chunks"])
        )

        response = bound_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        )

        # ── process tool calls ────────────────────────────────────────────────
        updates: Dict[str, Any] = {inbox_key: []}  # always clear own inbox

        for tc in getattr(response, "tool_calls", []):
            tool_result = tool_map[tc["name"]].invoke(tc["args"])

            if tc["name"].startswith("send_peer_query"):
                # Route outgoing message into target's inbox.
                # Preserve the debate thread and increment turn counter.
                target = tool_result["target_agent"]
                tool_result["debate_id"] = msg["debate_id"]
                tool_result["turn_number"] = msg["turn_number"] + 1

                updates[f"{target}_inbox"] = [tool_result]

                # Bump turn_count on the existing DebateThread.
                existing_thread = state["debate_threads"].get(msg["debate_id"], {})
                updates["debate_threads"] = {
                    msg["debate_id"]: {
                        **existing_thread,
                        "turn_count": msg["turn_number"] + 1,
                    }
                }

            elif tc["name"].startswith("resolve_debate"):
                # Mark thread resolved and surface the finding.
                did = tool_result["debate_id"]
                existing_thread = state["debate_threads"].get(did, {})
                updates["debate_threads"] = {
                    did: {
                        **existing_thread,
                        "is_resolved": True,
                        "resolution_summary": tool_result["resolution_summary"],
                    }
                }
                updates["resolved_findings"] = [tool_result]

        return updates

    agent_node.__name__ = f"{agent_name}_agent_node"
    return agent_node
