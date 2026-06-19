#!/usr/bin/env python3
"""
evaluate_hf_questions.py
────────────────────────
Runs the HuggingFace EnterpriseRAG-Bench test questions through the RAG Swarm
and evaluates RAG metrics (Context Precision, Context Recall, Faithfulness) using DeepEval and Gemini.
"""

import os
import uuid
import textwrap
from datasets import load_dataset

from module_a.pipeline import RAGPipeline
from module_d.leader_state import make_audit_task, make_initial_global_state
from module_d.leader_graph import build_leader_graph

# DeepEval imports for Spec §5.1 RAG Metrics
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel # Import the Gemini Model wrapper

def run_evaluation():
    # Ensure the Google API key is set in your environment
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("Please set your GOOGLE_API_KEY environment variable to use Gemini for DeepEval.")

    print("Loading test questions from HuggingFace...")
    ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
    questions = ds["test"]
    
    print(f"Loaded {len(questions)} test questions.")
    
    # Initialize our pipeline & agent swarm
    rag = RAGPipeline()
    leader_graph = build_leader_graph(use_sqlite_checkpointer=False)

    # Taking a small batch to demonstrate the pipeline and save API costs during testing
    test_batch = questions.select(range(min(3, len(questions))))
    test_cases = []
    
    print("\nPreparing tasks and generating Swarm responses for DeepEval...")
    for idx, row in enumerate(test_batch):
        q_id = row['question_id']
        question_text = row['question']
        gold_answer = row['gold_answer']
        
        print(f"\n[{q_id}] Retrieving context and processing: {question_text}")
        
        # 1. Retrieve Context via Module A
        retrieval_hits = rag.retrieve(question_text)
        
        # Format chunks for the Swarm dispatch
        evidence_chunks = [
            f"[Source: {hit.get('metadata', {}).get('filename', 'Unknown')}]\n{hit.get('document', '')}"
            for hit in retrieval_hits.retrieved_chunks
        ]
        
        # Format pure document strings for DeepEval Context Metrics
        retrieval_context = [hit.get('document', '') for hit in retrieval_hits.retrieved_chunks]
        
        # 2. Dispatch to Swarm
        task = make_audit_task(
            task_id=q_id,
            initiating_agent="financial",
            target_agent="legal",
            query=question_text,
            evidence_chunks=evidence_chunks
        )
        
        run_id = f"hf-eval-{uuid.uuid4().hex[:8]}"
        initial_state = make_initial_global_state(audit_tasks=[task], run_id=run_id)
        
        # Run the swarm for this single task to cleanly pair input/output
        result = leader_graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
        
        # 3. Extract the Swarm's final negotiated answer
        actual_output = "No resolution reached."
        for debate_id, finding in result.get("shared_memory", {}).items():
            actual_output = finding.get('resolution_summary', actual_output)
            break # Isolate the first resolution for this specific task
            
        print(f"Swarm Output: {textwrap.shorten(actual_output, width=100)}")
        
        # 4. Create DeepEval Test Case
        test_case = LLMTestCase(
            input=question_text,
            actual_output=actual_output,
            expected_output=gold_answer,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)
        
    print("\n" + "="*60 + "\nRUNNING DEEPEVAL METRICS (Spec §5.1) WITH GEMINI\n" + "="*60)
    
    # Initialize the Gemini model for DeepEval
    # You can change 'gemini-2.5-flash' to 'gemini-2.5-pro' if you want a more rigorous (but slower) judge
    gemini_judge = GeminiModel(model="gemini-2.5-flash", api_key=google_api_key)

    # Initialize the required RAG metrics, passing the gemini_judge to the 'model' parameter
    precision = ContextualPrecisionMetric(threshold=0.5, include_reason=True, model=gemini_judge)
    recall = ContextualRecallMetric(threshold=0.5, include_reason=True, model=gemini_judge)
    faithfulness = FaithfulnessMetric(threshold=0.5, include_reason=True, model=gemini_judge)
    
    # Execute the evaluation framework
    evaluate(test_cases, [precision, recall, faithfulness])
    
    print("\nEvaluation complete. DeepEval results and reasoning are outputted above.")

if __name__ == "__main__":
    run_evaluation()