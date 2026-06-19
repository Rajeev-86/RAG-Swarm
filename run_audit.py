#!/usr/bin/env python3
"""
run_audit.py
────────────
Main execution entry point for the Federated M&A Due Diligence Swarm.
Integrates Module A (RAG Layer), Module C (Mesh), and Module D (Leader).

Usage:
    python run_audit.py
"""

import sys
import uuid
import pprint
from typing import List

# Setup environment if needed
from dotenv import load_dotenv # if exists? Or just python standard

from module_a.pipeline import RAGPipeline
from module_d.leader_graph import build_leader_graph
from module_d.leader_state import make_audit_task, make_initial_global_state

load_dotenv()

def main():
    print("Initializing RAG Pipeline (Module A)...")
    rag = RAGPipeline()
    
    print("Building Leader Graph (Module D & C)...")
    leader_graph = build_leader_graph(use_sqlite_checkpointer=False)

    # Let's define some cross-domain audit tasks to kick off the swarm
    task_definitions = [
        {
            "initiating_agent": "financial",
            "target_agent": "legal",
            "query": "Do the contract penalty clauses cover the identified $12M revenue discrepancy?",
        },
        {
            "initiating_agent": "cybersecurity",
            "target_agent": "hr",
            "query": "Does the recent data breach report indicate any employee misconduct or require termination protocols?",
        }
    ]
    
    tasks = []
    print("\nPreparing Audit Tasks and retrieving initial context...")
    for i, td in enumerate(task_definitions):
        print(f"  [{i+1}/{len(task_definitions)}] Retrieving context for: {td['query']}")
        
        # Module A: 5-Stage RAG engine
        retrieval_result = rag.retrieve(td["query"])
        
        # Format the RAG results into chunk strings
        evidence_chunks = [
            f"[Source: {hit.get('metadata', {}).get('filename', 'Unknown')}]\n{hit.get('document', '')}"
            for hit in retrieval_result.retrieved_chunks
        ]
        
        task = make_audit_task(
            task_id=f"task_{i+1}",
            initiating_agent=td["initiating_agent"],
            target_agent=td["target_agent"],
            query=td["query"],
            evidence_chunks=evidence_chunks
        )
        tasks.append(task)
        
    run_id = f"audit-run-{uuid.uuid4().hex[:8]}"
    initial_state = make_initial_global_state(audit_tasks=tasks, run_id=run_id)
    
    print(f"\nKicking off Swarm Audit (Run ID: {run_id})...\n" + "="*60)
    
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
