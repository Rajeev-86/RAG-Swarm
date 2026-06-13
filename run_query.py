#!/usr/bin/env python3
"""
run_query.py
─────────────
Interactive query loop against an already-ingested data room.

Usage:
    python run_query.py
    python run_query.py "What are the key legal risks in this acquisition?"
"""

import sys
from module_a.pipeline import RAGPipeline

SINGLE_QUERY = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

SAMPLE_QUERIES = [
    "What change-of-control provisions trigger vesting acceleration?",
    "Summarise the cybersecurity vulnerabilities and their remediation cost.",
    "What are the revenue concentration risks and top client renewal concerns?",
    "Are there outstanding tax liabilities or GDPR compliance gaps?",
]

if __name__ == "__main__":
    pipeline = RAGPipeline()

    queries = [SINGLE_QUERY] if SINGLE_QUERY else SAMPLE_QUERIES

    for query in queries:
        print("\n" + "═" * 70)
        print(f"QUERY: {query}")
        print("═" * 70)
        result  = pipeline.retrieve(query)
        context = pipeline.format_context(result)
        print(context)