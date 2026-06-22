#!/usr/bin/env python3
"""
run_audit.py
────────────
Main execution entry point for the Federated M&A Due Diligence Swarm.
Integrates Module A (RAG Layer), Module B (Domain Agents), 
Module C (Mesh), and Module D (Leader).

Usage:
    python run_audit.py
"""

import sys
import uuid
import pprint
from typing import List

from dotenv import load_dotenv

# Import all integration points
from module_a.pipeline import RAGPipeline
from module_b.agent_factory import run_all_agents
from module_b.bridge import agent_results_to_audit_tasks
from module_d.leader_graph import build_leader_graph
from module_d.leader_state import make_initial_global_state

load_dotenv()

def main():
    print("Initializing RAG Pipeline (Module A)...")
    rag = RAGPipeline()
    
    print("Building Leader Graph (Module D & C)...")
    leader_graph = build_leader_graph(use_sqlite_checkpointer=False)

    # 1. Provide an overarching query to kick off the Due Diligence sweep
    overarching_query = (
        "Identify all critical legal, financial, HR, and cybersecurity risks in the "
        "target company's data room, paying special attention to the $12M revenue "
        "discrepancy and the recent data breach."
    )
    
    print(f"\n[Step 1] Module A: Retrieving context for the overarching sweep...")
    print(f"Query: {overarching_query}")
    retrieval_result = rag.retrieve(overarching_query)
    
    print(f"\n[Step 2] Module B: Running Domain Agents for initial extraction...")
    # Module B independently processes the chunks, generates Findings, 
    # and creates PeerQueries for anything requiring cross-domain validation
    agent_results = run_all_agents(retrieval_result)
    
    print("\n[Module B Findings Summary]")
    for domain, result in agent_results.items():
        print(f"  - {domain.value.upper()}: {len(result.findings)} finding(s), "
              f"{len(result.peer_queries)} peer query/queries flagged.")

    print(f"\n[Step 3] Bridge: Converting cross-domain PeerQueries to Swarm AuditTasks...")
    # The bridge merges duplicate queries and sets up the Swarm payload
    tasks = agent_results_to_audit_tasks(
        agent_results, 
        prioritise_critical=True, 
        deduplicate_pairs=True
    )
    
    # If Module B didn't flag any cross-domain issues, the Swarm doesn't need to run
    if not tasks:
        print("\nNo cross-domain peer queries were generated. Audit sweep complete. "
              "All risks were self-contained within their respective domains.")
        return
        
    run_id = f"audit-run-{uuid.uuid4().hex[:8]}"
    initial_state = make_initial_global_state(audit_tasks=tasks, run_id=run_id)
    
    print(f"\n[Step 4] Module C/D: Kicking off Swarm Audit (Run ID: {run_id})...\n" + "="*60)
    
    # Run the swarm
    result = leader_graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
    
    print("\n" + "="*60 + "\nAUDIT COMPLETE\n" + "="*60)
    
    print("\n[Data Room Status]")
    pprint.pprint(result.get("data_room_status"))
    
    print("\n[Shared Memory Findings]")
    for debate_id, finding in result.get("shared_memory", {}).items():
        resolved_by = finding.get("resolved_by", "unknown")
        print(f"\n  - Debate ID: {debate_id} (Resolved by: {resolved_by})")
        print(f"    Summary: {finding.get('resolution_summary')}")
        
    if result.get("last_checkpoint_path"):
        print(f"\n[!] A Kill-Switch event occurred. JSON Checkpoint saved at: {result['last_checkpoint_path']}")
        
if __name__ == "__main__":
    main()