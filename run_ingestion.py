#!/usr/bin/env python3
"""
run_ingestion.py
─────────────────
One-shot script to ingest all documents from your data room.

Usage:
    python run_ingestion.py                        # uses ./data/data_room
    python run_ingestion.py /path/to/data_room     # custom path
"""

import sys
from module_a.pipeline import RAGPipeline

DATA_ROOM = sys.argv[1] if len(sys.argv) > 1 else "./data/data_room"

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.ingest(DATA_ROOM)
    print(f"\n✓ Ingestion complete.")
    print(f"  ChromaDB chunks: {pipeline.vector_store.count()}")
    print(f"  BM25 documents:  {len(pipeline.bm25_store)}")