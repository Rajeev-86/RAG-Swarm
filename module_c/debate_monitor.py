"""
module_c/debate_monitor.py
──────────────────────────
Kill-switch detection and mesh routing logic.

debate_monitor_node()  — scans active threads, fires kill-switch if needed.
should_continue_mesh() — LangGraph conditional-edge function: loop or exit.

Kill-switch conditions (spec §3.2)
───────────────────────────────────
  1. Turn-limit breach   — any unresolved debate exceeds MAX_TURNS (>3).
  2. Context fragmentation — the same agent pair has >1 concurrent unresolved
     thread (circular debate loop detected).

When either condition fires the node writes kill_switch_triggered=True
plus a human-readable reason.  The Leader Agent (Module D) reads these
fields on mesh exit and decides whether to force-summarise or checkpoint.
"""
from __future__ import annotations

from typing import Any, Dict, Literal

from .mesh_state import MeshState

MAX_TURNS: int = 3          # spec §3.2: "P2P debate exceeds turn limits (>3)"
_DOMAIN_AGENTS = ("financial", "legal", "hr", "cyber")


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring node
# ─────────────────────────────────────────────────────────────────────────────

def debate_monitor_node(state: MeshState) -> Dict[str, Any]:
    """
    Passively scan all DebateThreads for kill-switch conditions.

    Called after every domain-agent turn so the Leader can intercept
    a runaway debate before context corruption propagates.

    Returns a partial state update.  Only sets kill_switch_triggered=True
    when a condition fires; otherwise explicitly clears it (False) so a
    stale flag from a previous tick doesn't linger.
    """
    threads = state.get("debate_threads", {})

    for debate_id, thread in threads.items():
        # Skip already-closed threads.
        if thread.get("is_resolved") or thread.get("is_escalated"):
            continue

        # ── Condition 1: turn-limit breach ───────────────────────────────────
        if thread.get("turn_count", 0) > MAX_TURNS:
            return {
                "kill_switch_triggered": True,
                "kill_switch_reason": (
                    f"Debate '{debate_id}' between "
                    f"'{thread['initiating_agent']}' ↔ '{thread['responding_agent']}' "
                    f"exceeded the {MAX_TURNS}-turn limit "
                    f"(current turns: {thread['turn_count']})."
                ),
                "kill_switch_debate_id": debate_id,
            }

    # ── Condition 2: context fragmentation ───────────────────────────────────
    # Build a list of sorted agent-pairs from all active (unresolved) threads.
    seen_pairs: list = []
    for thread in threads.values():
        if thread.get("is_resolved") or thread.get("is_escalated"):
            continue
        pair = tuple(sorted([thread["initiating_agent"], thread["responding_agent"]]))
        if pair in seen_pairs:
            return {
                "kill_switch_triggered": True,
                "kill_switch_reason": (
                    f"Context fragmentation: agents '{pair[0]}' and '{pair[1]}' "
                    f"have multiple concurrent unresolved debate threads."
                ),
                "kill_switch_debate_id": None,
            }
        seen_pairs.append(pair)

    # ── All clear ─────────────────────────────────────────────────────────────
    return {"kill_switch_triggered": False}


# ─────────────────────────────────────────────────────────────────────────────
# Routing function
# ─────────────────────────────────────────────────────────────────────────────

def should_continue_mesh(
    state: MeshState,
) -> Literal["continue_mesh", "escalate_to_leader"]:
    """
    LangGraph conditional-edge function attached to debate_monitor.

    Decision logic:
      escalate  ← kill switch fired
      escalate  ← all threads resolved (clean exit, findings ready)
      escalate  ← no threads at all
      escalate  ← open threads exist but every inbox is empty
                  (agents replied without spawning new queries → stall)
      continue  ← at least one open thread AND at least one pending inbox
    """
    if state.get("kill_switch_triggered"):
        return "escalate_to_leader"

    threads = state.get("debate_threads", {})

    if not threads:
        return "escalate_to_leader"

    all_closed = all(
        t.get("is_resolved") or t.get("is_escalated")
        for t in threads.values()
    )
    if all_closed:
        return "escalate_to_leader"

    any_pending = any(
        state.get(f"{agent}_inbox", []) for agent in _DOMAIN_AGENTS
    )
    return "continue_mesh" if any_pending else "escalate_to_leader"
