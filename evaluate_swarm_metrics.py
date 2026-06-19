#!/usr/bin/env python3
"""
evaluate_swarm_metrics.py
─────────────────────────
Evaluates the non-deterministic agentic trajectories of the RAG Swarm
using DeepEval's GEval (LLM-as-a-Judge) powered by Google Gemini.
Scores Spec §5.2 criteria: Tool Correctness, Step Efficiency, and Plan Adherence.
"""

import os
import json
import uuid

# DeepEval imports for LLM-as-a-Judge custom metrics
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel

from module_d.leader_state import make_audit_task, make_initial_global_state
from module_d.leader_graph import build_leader_graph

def run_swarm_evaluation():
    # Setup Gemini as the Judge
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("Please set your GOOGLE_API_KEY environment variable.")
        
    # Using gemini-2.5-pro for higher reasoning accuracy during trajectory evaluation
    gemini_judge = GeminiModel(model="gemini-2.5-pro", api_key=google_api_key)

    print("Initializing Swarm...")
    leader_graph = build_leader_graph(use_sqlite_checkpointer=False)

    # Inject a complex cross-domain query that forces Legal and Financial to debate
    complex_query = "Do the change-of-control penalty clauses cover the $12M revenue discrepancy, and what are the exact indemnification caps?"
    mock_evidence = [
        "[Source: acquisition_agreement.txt] Section 12.3: The Target Company shall pay a termination fee of $18.5M if this Agreement is terminated...",
        "[Source: financial_audit_2023.txt] Client #1 (Acme Corp): $31.2M (25% of revenue) — renewal risk flagged. EBITDA margin improved due to one-time $8.5M insurance recovery."
    ]

    # Dispatch the task
    task = make_audit_task(
        task_id="eval_swarm_task_1",
        initiating_agent="financial",
        target_agent="legal",
        query=complex_query,
        evidence_chunks=mock_evidence
    )

    run_id = f"swarm-eval-{uuid.uuid4().hex[:8]}"
    initial_state = make_initial_global_state(audit_tasks=[task], run_id=run_id)

    print(f"Executing swarm trajectory for Run ID: {run_id}...")
    result = leader_graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})

    # Extract Trajectory Data (The chronological event log and the final findings)
    audit_log = result.get("audit_log", [])
    shared_memory = result.get("shared_memory", {})
    
    # Serialize the execution trajectory into text for the LLM judge
    trajectory_dump = json.dumps({
        "audit_log": audit_log,
        "shared_memory": shared_memory
    }, indent=2)

    print(f"Trajectory captured. Log events: {len(audit_log)}, Findings: {len(shared_memory)}")

    # ── Define Spec §5.2 Metrics using GEval ──────────────────────────────────
    
    tool_correctness = GEval(
        name="Tool Correctness",
        criteria="Analyze the trajectory. Determine whether the agents formatted and passed arguments correctly during P2P handoffs (e.g., valid target_agent, evidence_chunks, debate_id). Agents should not hallucinate tool inputs.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gemini_judge,
        threshold=0.7
    )

    step_efficiency = GEval(
        name="Step Efficiency",
        criteria="Analyze the trajectory. Determine if the agents resolved adversarial disputes efficiently. The mesh limits P2P debates to a maximum of 3 turns. Verify if tasks were completed in under 3 turns, or if the Leader Agent successfully hit the kill-switch when the limit was exceeded.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gemini_judge,
        threshold=0.7
    )

    plan_adherence = GEval(
        name="Plan Adherence",
        criteria="Analyze the trajectory's audit_log. Determine if the Leader Agent successfully managed the overall execution flow. Look for the exact proper sequence of events: AUDIT_STARTED -> MESH_DISPATCHED -> either MESH_COMPLETE or KILL_SWITCH_FIRED -> TASK_ADVANCED -> AUDIT_COMPLETE.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gemini_judge,
        threshold=1.0 # Strict adherence required; flow must match perfectly
    )

    # ── Execute Evaluation ───────────────────────────────────────────────────
    
    # We pass the entire JSON trajectory as the 'actual_output' so GEval can read it
    test_case = LLMTestCase(
        input="Evaluate the non-deterministic agentic trajectory of the swarm for this cross-domain M&A task.",
        actual_output=trajectory_dump
    )

    print("\n" + "="*70 + "\nRUNNING DEEPEVAL SWARM METRICS (Spec §5.2) WITH GEMINI\n" + "="*70)
    evaluate([test_case], [tool_correctness, step_efficiency, plan_adherence])

if __name__ == "__main__":
    run_swarm_evaluation()