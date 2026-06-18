"""
module_d/tests/test_leader.py
──────────────────────────────
Comprehensive unit and integration test suite for the Leader Agent (Module D).

Coverage:
  ✓ State bootstrapper and checkpoint fallbacks
  ✓ Routing logic (clean exit vs. kill-switch)
  ✓ All individual node functions (handle_mesh_result, force_summary, checkpoint, etc.)
  ✓ Full StateGraph execution paths via .invoke() using a mocked Mesh graph
"""
import json
import os
import uuid
from unittest.mock import MagicMock
import pytest

from module_d.leader_state import (
    make_audit_task,
    make_initial_global_state,
    make_log_entry,
    GlobalState
)
from module_d.checkpoint_manager import get_checkpointer

from module_d.leader_nodes import (
    handle_mesh_result_node,
    build_force_summary_node,
    build_checkpoint_node,
    reset_context_node,
    advance_task_node,
    check_complete_node,
    build_dispatch_mesh_node
)
from module_d.leader_graph import (
    build_leader_graph,
    _route_after_mesh_result,
    _route_after_advance_task,
    _route_after_check_complete
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. TRIVIAL / EXISTING TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_make_audit_task():
    task = make_audit_task("t1", "financial", "legal", "q", ["chunk1"])
    assert task["task_id"] == "t1"
    assert task["initiating_agent"] == "financial"
    assert task["target_agent"] == "legal"
    assert task["status"] == "pending"

def test_make_initial_global_state():
    task = make_audit_task("t1", "financial", "legal", "q", ["chunk1"])
    state = make_initial_global_state([task], "run_1")
    assert state["run_id"] == "run_1"
    assert state["current_task_idx"] == 0
    assert len(state["audit_log"]) == 1
    assert "financial" in state["data_room_status"]
    assert "legal" in state["data_room_status"]
    assert not state["audit_complete"]

def test_memory_checkpointer_fallback():
    cp = get_checkpointer(checkpoint_dir="dummy")
    assert cp is not None


# ═════════════════════════════════════════════════════════════════════════════
# 2. ROUTING FUNCTION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_route_after_mesh_result():
    # Clean Path
    state_clean = {"last_mesh_state": {"kill_switch_triggered": False}}
    assert _route_after_mesh_result(state_clean) == "advance_task"

    # Kill-Switch Path
    state_ks = {"last_mesh_state": {"kill_switch_triggered": True}}
    assert _route_after_mesh_result(state_ks) == "force_summary"

def test_route_after_advance_task():
    # Clean Path -> Check Complete
    state_clean = {"last_mesh_state": {"kill_switch_triggered": False}}
    assert _route_after_advance_task(state_clean) == "check_complete"

    # Kill-Switch Path -> Wipe context
    state_ks = {"last_mesh_state": {"kill_switch_triggered": True}}
    assert _route_after_advance_task(state_ks) == "reset_context"

def test_route_after_check_complete():
    assert _route_after_check_complete({"audit_complete": True}) == "__end__"
    assert _route_after_check_complete({"audit_complete": False}) == "dispatch_mesh"


# ═════════════════════════════════════════════════════════════════════════════
# 3. INDIVIDUAL NODE TESTS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_state():
    task = make_audit_task("t1", "financial", "legal", "test", [])
    return make_initial_global_state([task], "test_run")

def test_handle_mesh_result_node_clean(base_state):
    base_state["last_mesh_state"] = {
        "kill_switch_triggered": False,
        "resolved_findings": [{"debate_id": "d1", "resolution_summary": "ok"}]
    }
    
    updates = handle_mesh_result_node(base_state)
    assert "d1" in updates["shared_memory"]
    assert updates["shared_memory"]["d1"]["resolution_summary"] == "ok"
    assert "MESH_COMPLETE" in updates["audit_log"][0]["event_type"]

def test_handle_mesh_result_node_kill_switch(base_state):
    base_state["last_mesh_state"] = {
        "kill_switch_triggered": True,
        "kill_switch_reason": "timeout",
        "resolved_findings": [{"debate_id": "d1", "resolution_summary": "partial"}]
    }
    
    updates = handle_mesh_result_node(base_state)
    # Shouldn't absorb findings directly on kill_switch
    assert "shared_memory" not in updates 
    assert "KILL_SWITCH_FIRED" in updates["audit_log"][0]["event_type"]

def test_force_summary_node_with_mocked_llm(base_state):
    base_state["last_mesh_state"] = {
        "kill_switch_triggered": True,
        "kill_switch_debate_id": "debate_123"
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Forced summary text."
    
    node = build_force_summary_node(llm=mock_llm)
    updates = node(base_state)
    
    assert "debate_123" in updates["shared_memory"]
    finding = updates["shared_memory"]["debate_123"]
    assert finding["is_forced"] is True
    assert finding["resolved_by"] == "leader"
    assert finding["resolution_summary"] == "Forced summary text."
    assert "FORCE_SUMMARY_GENERATED" in updates["audit_log"][0]["event_type"]

def test_checkpoint_node_writes_to_disk(base_state, tmp_path):
    base_state["last_mesh_state"] = {"kill_switch_reason": "test trigger"}
    
    node = build_checkpoint_node(checkpoint_dir=str(tmp_path))
    updates = node(base_state)
    
    assert updates["checkpoint_count"] == 1
    assert updates["last_checkpoint_path"].endswith(".json")
    assert os.path.exists(updates["last_checkpoint_path"])
    
    with open(updates["last_checkpoint_path"], "r") as f:
        data = json.load(f)
        assert data["meta"]["run_id"] == "test_run"
        assert data["meta"]["trigger"] == "test trigger"

def test_advance_task_node_clean_status(base_state):
    base_state["last_mesh_state"] = {"kill_switch_triggered": False}
    
    updates = advance_task_node(base_state)
    assert updates["audit_tasks"][0]["status"] == "complete"
    assert updates["current_task_idx"] == 1
    assert updates["data_room_status"]["financial"]["tasks_complete"] == 1
    assert updates["data_room_status"]["financial"]["tasks_force_resolved"] == 0

def test_advance_task_node_kill_switch_status(base_state):
    base_state["last_mesh_state"] = {"kill_switch_triggered": True}
    
    updates = advance_task_node(base_state)
    assert updates["audit_tasks"][0]["status"] == "force_resolved"
    assert updates["current_task_idx"] == 1
    assert updates["data_room_status"]["financial"]["tasks_force_resolved"] == 1
    assert updates["data_room_status"]["financial"]["has_escalations"] is True

def test_reset_context_node(base_state):
    base_state["last_mesh_state"] = {"data": "should_be_wiped"}
    updates = reset_context_node(base_state)
    assert updates["last_mesh_state"] is None

def test_check_complete_node(base_state):
    # Incomplete
    updates = check_complete_node(base_state)
    assert updates["audit_complete"] is False

    # Complete
    base_state["current_task_idx"] = 1
    updates_done = check_complete_node(base_state)
    assert updates_done["audit_complete"] is True
    assert "AUDIT_COMPLETE" in updates_done["audit_log"][0]["event_type"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. FULL GRAPH INTEGRATION TESTS (MOCKED MESH)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_integration_state():
    tasks = [
        make_audit_task("t1", "financial", "legal", "Query 1", []),
        make_audit_task("t2", "hr", "cyber", "Query 2", [])
    ]
    return make_initial_global_state(tasks, run_id="integ-run-001")

def test_end_to_end_clean_path(base_integration_state, tmp_path):
    """
    Simulates a full end-to-end run where Module C returns successfully every time.
    """
    mock_mesh_graph = MagicMock()
    # Mocking what the mesh returns for a CLEAN exit
    mock_mesh_graph.invoke.return_value = {
        "kill_switch_triggered": False,
        "resolved_findings": [{"debate_id": "d1", "resolution_summary": "clean resolution"}],
        "mesh_tick": 2
    }
    
    graph = build_leader_graph(
        mesh_graph=mock_mesh_graph,
        checkpoint_dir=str(tmp_path),
        use_sqlite_checkpointer=False,
        _mesh_state_factory=MagicMock(return_value={"seed": "state"})
    )
    
    result = graph.invoke(base_integration_state, config={"configurable": {"thread_id": "clean-run"}})
    
    # Assertions
    assert result["audit_complete"] is True
    assert result["current_task_idx"] == 2
    assert result["audit_tasks"][0]["status"] == "complete"
    assert result["audit_tasks"][1]["status"] == "complete"
    
    # Since it was clean, the checkpoints count should be 0
    assert result["checkpoint_count"] == 0
    
    # Two tasks -> two clean resolutions -> one finding per task in shared memory (overwritten/merged since mock returns same id)
    assert "d1" in result["shared_memory"]
    assert result["shared_memory"]["d1"]["resolution_summary"] == "clean resolution"

def test_end_to_end_kill_switch_path(base_integration_state, tmp_path):
    """
    Simulates a full end-to-end run where Module C triggers the kill-switch on every task.
    """
    mock_mesh_graph = MagicMock()
    # Mocking what the mesh returns for a KILL-SWITCH exit
    mock_mesh_graph.invoke.return_value = {
        "kill_switch_triggered": True,
        "kill_switch_reason": "Mocked timeout fragmentation",
        "kill_switch_debate_id": "killed_d1",
        "resolved_findings": [{"debate_id": "killed_d1", "resolution_summary": "partial"}],
        "mesh_tick": 4
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Forced summary synthesis."

    graph = build_leader_graph(
        llm=mock_llm,
        mesh_graph=mock_mesh_graph,
        checkpoint_dir=str(tmp_path),
        use_sqlite_checkpointer=False,
        _mesh_state_factory=MagicMock(return_value={"seed": "state"})
    )

    result = graph.invoke(base_integration_state, config={"configurable": {"thread_id": "ks-run"}})

    # Assertions
    assert result["audit_complete"] is True
    assert result["current_task_idx"] == 2
    
    # Verify tasks were forced resolved
    assert result["audit_tasks"][0]["status"] == "force_resolved"
    assert result["audit_tasks"][1]["status"] == "force_resolved"
    
    # Because 2 tasks were kill-switched, we should have 2 checkpoints written to disk
    assert result["checkpoint_count"] == 2
    assert os.path.exists(result["last_checkpoint_path"])
    
    # The forced finding should be in shared memory
    assert "killed_d1" in result["shared_memory"]
    assert result["shared_memory"]["killed_d1"]["is_forced"] is True
    assert result["shared_memory"]["killed_d1"]["resolution_summary"] == "Forced summary synthesis."