#!/usr/bin/env python3
"""
evaluate_hf_questions.py
────────────────────────
Runs the HuggingFace EnterpriseRAG-Bench test questions through the RAG Swarm.
"""

from datasets import load_dataset
from module_a.pipeline import RAGPipeline
from module_d.leader_state import make_audit_task, make_initial_global_state
from module_d.leader_graph import build_leader_graph
import uuid
import pprint

def run_evaluation():
    print("Loading test questions from HuggingFace...")
    ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
    questions = ds["test"]
    
    print(f"Loaded {len(questions)} test questions.")
    
    # Initialize our pipeline & agent swarm
    rag = RAGPipeline()
    leader_graph = build_leader_graph(use_sqlite_checkpointer=False)

    # Let's take the first 3 questions just to demonstrate the pipeline
    test_batch = questions.select(range(min(3, len(questions))))
    tasks = []
    
    print("\nPreparing tasks for Swarm dispatch...")
    for idx, row in enumerate(test_batch):
        q_id = row['question_id']
        question_text = row['question']
        
        print(f"\n[{q_id}] Retrieving context for: {question_text}")
        
        # Pull context from our Module A data layer
        retrieval_hits = rag.retrieve(question_text)
        evidence_chunks = [hit.content for hit in retrieval_hits]
        
        # We can dynamically assign domains based on the question_type if needed, 
        # but defaulting to financial <-> legal for demonstration.
        task = make_audit_task(
            task_id=q_id,
            initiating_agent="financial",
            target_agent="legal",
            query=question_text,
            evidence_chunks=evidence_chunks
        )
        tasks.append(task)
        
    run_id = f"hf-eval-{uuid.uuid4().hex[:8]}"
    initial_state = make_initial_global_state(audit_tasks=tasks, run_id=run_id)
    
    print(f"\nKicking off Swarm on {len(tasks)} questions... (Run ID: {run_id})")
    result = leader_graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
    
    print("\n" + "="*60 + "\nEVALUATION COMPLETE\n" + "="*60)
    for q_idx in range(len(test_batch)):
        q_id = test_batch[q_idx]['question_id']
        gold = test_batch[q_idx]['gold_answer']
        
        # The debate finding is keyed by its debate_id. Wait, how do we get the debate_id?
        # The task maps to an initiating mesh state. 
        # For simplicity in testing, we can just print the shared_memory findings overall.
        
    print("\n[Swarm Findings Generated]")
    for debate_id, finding in result.get("shared_memory", {}).items():
        print(f"\n[Generated Summary (Debate ID: {debate_id})]:")
        ans = finding.get('resolution_summary', '')
        # Simple text wrapping
        import textwrap
        print(textwrap.fill(ans, width=80))

if __name__ == "__main__":
    run_evaluation()
