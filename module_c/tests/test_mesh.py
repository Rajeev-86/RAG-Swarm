"""
module_c/tests/test_mesh.py
───────────────────────────
Unit tests for Module C — Mesh Interface (P2P Debate Layer).

Run:  pytest module_c/tests/ -v

Coverage
────────
  1. Peer tools         — send_peer_query, resolve_debate tool factories
  2. Debate monitor     — kill-switch conditions (turn limit, fragmentation)
  3. Routing function   — should_continue_mesh decision branches
  4. State bootstrapper — create_initial_mesh_state
  5. Agent node wiring  — build_agent_node with a mocked LLM (no Ollama needed)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from module_c.debate_monitor import MAX_TURNS, debate_monitor_node, should_continue_mesh
from module_c.mesh_graph import create_initial_mesh_state
from module_c.mesh_state import DebateThread, MeshState, PeerMessage
from module_c.peer_tools import create_resolve_debate_tool, create_send_peer_query_tool


# ─────────────────────────────────────────────────────────────────────────────
# Shared test factories
# ─────────────────────────────────────────────────────────────────────────────

def _msg(
    source: str = "financial",
    target: str = "legal",
    turn: int = 1,
    debate_id: str | None = None,
) -> PeerMessage:
    return {
        "message_id": str(uuid.uuid4()),
        "debate_id": debate_id or str(uuid.uuid4()),
        "source_agent": source,
        "target_agent": target,
        "query": "Are penalty clauses triggered by the $12 M revenue gap?",
        "evidence_chunks": ["§4.2 APA: penalty threshold 5 %."],
        "turn_number": turn,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _thread(
    debate_id: str | None = None,
    turn_count: int = 1,
    is_resolved: bool = False,
    is_escalated: bool = False,
    initiating: str = "financial",
    responding: str = "legal",
) -> DebateThread:
    return {
        "debate_id": debate_id or str(uuid.uuid4()),
        "initiating_agent": initiating,
        "responding_agent": responding,
        "turn_count": turn_count,
        "is_resolved": is_resolved,
        "resolution_summary": None,
        "is_escalated": is_escalated,
    }


def _state(
    threads: Dict[str, DebateThread] | None = None,
    kill_switch: bool = False,
    inboxes: dict | None = None,
) -> MeshState:
    s: MeshState = {
        "financial_inbox": [],
        "legal_inbox": [],
        "hr_inbox": [],
        "cybersecurity_inbox": [],
        "debate_threads": threads or {},
        "active_agents": [],
        "mesh_tick": 0,
        "kill_switch_triggered": kill_switch,
        "kill_switch_reason": None,
        "kill_switch_debate_id": None,
        "resolved_findings": [],
    }
    if inboxes:
        s.update(inboxes)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 1. Peer tool tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSendPeerQueryTool:
    def test_valid_returns_peer_message_fields(self):
        tool = create_send_peer_query_tool("financial")
        result = tool.invoke(
            {
                "target_agent": "legal",
                "query": "Do penalty clauses cover the gap?",
                "evidence_chunks": ["§4.2 APA"],
            }
        )
        assert result["source_agent"] == "financial"
        assert result["target_agent"] == "legal"
        assert result["turn_number"] == 1
        assert "message_id" in result and uuid.UUID(result["message_id"])
        assert "debate_id" in result and uuid.UUID(result["debate_id"])
        assert "timestamp" in result

    def test_rejects_self_query(self):
        tool = create_send_peer_query_tool("financial")
        with pytest.raises(ValueError, match="cannot send a peer query to itself"):
            tool.invoke(
                {"target_agent": "financial", "query": "x", "evidence_chunks": []}
            )

    def test_rejects_unknown_agent(self):
        tool = create_send_peer_query_tool("financial")
        with pytest.raises(ValueError, match="Unknown agent"):
            tool.invoke(
                {"target_agent": "marketing", "query": "x", "evidence_chunks": []}
            )

    def test_continues_existing_thread(self):
        tool = create_send_peer_query_tool("legal")
        existing_id = str(uuid.uuid4())
        result = tool.invoke(
            {
                "target_agent": "financial",
                "query": "§4.2 caps liability at $5 M, not $12 M.",
                "evidence_chunks": ["§4.2 APA: cap = $5 M"],
                "debate_id": existing_id,
                "turn_number": 2,
            }
        )
        assert result["debate_id"] == existing_id
        assert result["turn_number"] == 2

    def test_all_four_agents_can_call_each_other(self):
        pairs = [
            ("financial", "legal"), ("legal", "financial"),
            ("hr", "cybersecurity"),        ("cybersecurity", "hr"),
        ]
        for source, target in pairs:
            tool = create_send_peer_query_tool(source)
            result = tool.invoke(
                {"target_agent": target, "query": "test", "evidence_chunks": []}
            )
            assert result["source_agent"] == source
            assert result["target_agent"] == target


class TestResolveDebateTool:
    def test_emits_correct_fields(self):
        tool = create_resolve_debate_tool("legal")
        did = str(uuid.uuid4())
        result = tool.invoke(
            {
                "debate_id": did,
                "resolution_summary": "§4.2 caps penalty at $5 M; confirmed.",
            }
        )
        assert result["debate_id"] == did
        assert result["resolved_by"] == "legal"
        assert "timestamp" in result

    def test_resolved_by_reflects_source_agent(self):
        for agent in ("financial", "legal", "hr", "cybersecurity"):
            tool = create_resolve_debate_tool(agent)
            result = tool.invoke(
                {"debate_id": str(uuid.uuid4()), "resolution_summary": "agreed"}
            )
            assert result["resolved_by"] == agent


# ─────────────────────────────────────────────────────────────────────────────
# 2. Debate monitor / kill-switch tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDebateMonitorNode:
    def test_no_kill_switch_at_max_turns(self):
        """Exactly at the limit — should NOT fire (spec says >MAX_TURNS)."""
        did = str(uuid.uuid4())
        result = debate_monitor_node(_state({did: _thread(did, turn_count=MAX_TURNS)}))
        assert result.get("kill_switch_triggered") is not True

    def test_kill_switch_fires_one_over_limit(self):
        did = str(uuid.uuid4())
        result = debate_monitor_node(
            _state({did: _thread(did, turn_count=MAX_TURNS + 1)})
        )
        assert result["kill_switch_triggered"] is True
        assert result["kill_switch_debate_id"] == did
        assert "exceeded" in result["kill_switch_reason"]
        assert str(MAX_TURNS) in result["kill_switch_reason"]

    def test_kill_switch_fires_on_fragmentation(self):
        """Same agent pair in two concurrent unresolved threads."""
        did1, did2 = str(uuid.uuid4()), str(uuid.uuid4())
        threads = {
            did1: _thread(did1, initiating="financial", responding="legal"),
            did2: _thread(did2, initiating="legal", responding="financial"),
        }
        result = debate_monitor_node(_state(threads))
        assert result["kill_switch_triggered"] is True
        assert "fragmentation" in result["kill_switch_reason"]

    def test_resolved_thread_not_evaluated(self):
        did = str(uuid.uuid4())
        result = debate_monitor_node(
            _state({did: _thread(did, turn_count=999, is_resolved=True)})
        )
        assert result.get("kill_switch_triggered") is not True

    def test_escalated_thread_not_evaluated(self):
        did = str(uuid.uuid4())
        result = debate_monitor_node(
            _state({did: _thread(did, turn_count=999, is_escalated=True)})
        )
        assert result.get("kill_switch_triggered") is not True

    def test_empty_threads_returns_clear(self):
        result = debate_monitor_node(_state({}))
        assert result.get("kill_switch_triggered") is False

    def test_different_pairs_do_not_trigger_fragmentation(self):
        """financial↔legal and hr↔cybersecurity are distinct pairs — no fragmentation."""
        did1, did2 = str(uuid.uuid4()), str(uuid.uuid4())
        threads = {
            did1: _thread(did1, initiating="financial", responding="legal"),
            did2: _thread(did2, initiating="hr", responding="cybersecurity"),
        }
        result = debate_monitor_node(_state(threads))
        assert result.get("kill_switch_triggered") is not True


# ─────────────────────────────────────────────────────────────────────────────
# 3. should_continue_mesh routing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldContinueMesh:
    def test_escalates_when_kill_switch_set(self):
        assert should_continue_mesh(_state(kill_switch=True)) == "escalate_to_leader"

    def test_escalates_when_no_threads(self):
        assert should_continue_mesh(_state()) == "escalate_to_leader"

    def test_escalates_when_all_resolved(self):
        did = str(uuid.uuid4())
        assert (
            should_continue_mesh(_state({did: _thread(did, is_resolved=True)}))
            == "escalate_to_leader"
        )

    def test_escalates_when_open_thread_but_no_pending_mail(self):
        """Open debate but all inboxes empty → agents reached consensus or stalled."""
        did = str(uuid.uuid4())
        assert (
            should_continue_mesh(_state({did: _thread(did, is_resolved=False)}))
            == "escalate_to_leader"
        )

    def test_continues_when_open_thread_and_pending_inbox(self):
        did = str(uuid.uuid4())
        msg = _msg(debate_id=did)
        state = _state(
            {did: _thread(did, is_resolved=False)},
            inboxes={"legal_inbox": [msg]},
        )
        assert should_continue_mesh(state) == "continue_mesh"

    def test_escalates_when_only_escalated_threads_remain(self):
        did = str(uuid.uuid4())
        msg = _msg(debate_id=did)
        state = _state(
            {did: _thread(did, is_escalated=True)},
            inboxes={"legal_inbox": [msg]},   # mail present but thread is closed
        )
        assert should_continue_mesh(state) == "escalate_to_leader"


# ─────────────────────────────────────────────────────────────────────────────
# 4. create_initial_mesh_state bootstrapper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateInitialMeshState:
    def test_seeds_target_inbox(self):
        state = create_initial_mesh_state(
            "financial", "legal", "Do penalty clauses cover the gap?", ["§4.2"]
        )
        assert len(state["legal_inbox"]) == 1
        msg = state["legal_inbox"][0]
        assert msg["source_agent"] == "financial"
        assert msg["target_agent"] == "legal"
        assert msg["turn_number"] == 1

    def test_other_inboxes_are_empty(self):
        state = create_initial_mesh_state("hr", "cybersecurity", "q", [])
        assert state["financial_inbox"] == []
        assert state["legal_inbox"] == []
        assert state["hr_inbox"] == []       # initiator, not receiver
        assert len(state["cybersecurity_inbox"]) == 1

    def test_creates_one_debate_thread(self):
        state = create_initial_mesh_state("financial", "legal", "q", [])
        assert len(state["debate_threads"]) == 1

    def test_thread_metadata_is_correct(self):
        state = create_initial_mesh_state("financial", "legal", "q", [])
        thread = next(iter(state["debate_threads"].values()))
        assert thread["initiating_agent"] == "financial"
        assert thread["responding_agent"] == "legal"
        assert thread["turn_count"] == 1
        assert thread["is_resolved"] is False
        assert thread["is_escalated"] is False

    def test_kill_switch_starts_clear(self):
        state = create_initial_mesh_state("financial", "legal", "q", [])
        assert state["kill_switch_triggered"] is False
        assert state["kill_switch_reason"] is None

    def test_raises_on_same_agent(self):
        with pytest.raises(ValueError, match="must differ"):
            create_initial_mesh_state("legal", "legal", "q", [])

    def test_debate_id_consistent_across_inbox_and_thread(self):
        state = create_initial_mesh_state("financial", "legal", "q", ["chunk"])
        thread_id = next(iter(state["debate_threads"]))
        msg_id = state["legal_inbox"][0]["debate_id"]
        assert thread_id == msg_id


# ─────────────────────────────────────────────────────────────────────────────
# 5. Agent node wiring test (mock LLM — no Ollama required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentNodeWiring:
    """
    Tests build_agent_node() with a mock LLM that simulates:
      (a) calling send_peer_query  → message routed to target inbox
      (b) calling resolve_debate   → thread marked resolved
      (c) returning no tool calls  → own inbox cleared, no side-effects
    """

    def _make_mock_llm(self, tool_call_name: str | None, tool_args: dict):
        """Build a mock bound LLM that returns a single tool call."""
        mock_response = MagicMock()
        if tool_call_name:
            mock_response.tool_calls = [{"name": tool_call_name, "args": tool_args}]
        else:
            mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_send_peer_query_routes_to_target_inbox(self):
        from module_c.agent_nodes import build_agent_node

        did = str(uuid.uuid4())
        mock_llm = self._make_mock_llm(
            "send_peer_query_from_financial",
            {
                "target_agent": "legal",
                "query": "Does §4.2 cap liability?",
                "evidence_chunks": ["§4.2 excerpt"],
                "debate_id": did,
                "turn_number": 1,
            },
        )

        node = build_agent_node("financial", llm=mock_llm)
        incoming_msg = _msg("hr", "financial", turn=1, debate_id=did)
        state = _state(
            {did: _thread(did, initiating="hr", responding="financial")},
            inboxes={"financial_inbox": [incoming_msg]},
        )

        updates = node(state)

        assert updates.get("financial_inbox") == []     # own inbox cleared
        assert "legal_inbox" in updates
        assert updates["legal_inbox"][0]["target_agent"] == "legal"
        assert updates["legal_inbox"][0]["turn_number"] == 2  # incremented

    def test_resolve_debate_marks_thread_resolved(self):
        from module_c.agent_nodes import build_agent_node

        did = str(uuid.uuid4())
        mock_llm = self._make_mock_llm(
            "resolve_debate_from_legal",
            {
                "debate_id": did,
                "resolution_summary": "§4.2 cap confirmed at $5 M.",
            },
        )

        node = build_agent_node("legal", llm=mock_llm)
        incoming_msg = _msg("financial", "legal", turn=2, debate_id=did)
        state = _state(
            {did: _thread(did, initiating="financial", responding="legal")},
            inboxes={"legal_inbox": [incoming_msg]},
        )

        updates = node(state)

        assert updates["legal_inbox"] == []
        assert updates["debate_threads"][did]["is_resolved"] is True
        assert len(updates["resolved_findings"]) == 1
        assert "cap confirmed" in updates["resolved_findings"][0]["resolution_summary"]

    def test_no_tool_call_clears_inbox_only(self):
        from module_c.agent_nodes import build_agent_node

        did = str(uuid.uuid4())
        mock_llm = self._make_mock_llm(None, {})   # no tool call

        node = build_agent_node("hr", llm=mock_llm)
        incoming_msg = _msg("financial", "hr", turn=1, debate_id=did)
        state = _state(
            {did: _thread(did, initiating="financial", responding="hr")},
            inboxes={"hr_inbox": [incoming_msg]},
        )

        updates = node(state)

        assert updates.get("hr_inbox") == []           # inbox cleared
        assert "financial_inbox" not in updates        # no outgoing message
        assert "resolved_findings" not in updates

    def test_idle_agent_returns_empty_clear(self):
        from module_c.agent_nodes import build_agent_node

        mock_llm = self._make_mock_llm(None, {})
        node = build_agent_node("cybersecurity", llm=mock_llm)
        state = _state()   # all inboxes empty

        updates = node(state)

        assert updates == {"cybersecurity_inbox": []}
        mock_llm.invoke.assert_not_called()   # LLM never invoked for idle agent
