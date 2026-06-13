import pytest
from module_d.leader_state import make_audit_task, make_initial_global_state, make_log_entry
from module_d.checkpoint_manager import get_checkpointer

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
