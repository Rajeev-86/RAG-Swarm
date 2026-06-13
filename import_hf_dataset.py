#!/usr/bin/env python3
"""
import_hf_dataset.py
────────────────────
Downloads the Onyx EnterpriseRAG-Bench dataset and populates the local data_room.
It converts remote documents into local Markdown files, retaining their metadata.
"""

import os
from datasets import load_dataset

DATA_ROOM_DIR = "data/data_room"

def import_documents():
    os.makedirs(DATA_ROOM_DIR, exist_ok=True)
    print("Downloading documents from HuggingFace (onyx-dot-app/EnterpriseRAG-Bench)...")
    
    # The dataset uses "test" split in the example
    ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents")
    documents = ds["test"]
    
    print(f"Downloaded {len(documents)} documents. Writing files to '{DATA_ROOM_DIR}/'...")
    
    for doc in documents:
        doc_id = doc.get("doc_id", "unknown_id")
        source_type = doc.get("source_type", "unknown_source")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        
        # Make a filesystem-safe filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:50].strip()
        filename = f"{source_type}_{doc_id}_{safe_title}.md".replace("/", "_")
        filepath = os.path.join(DATA_ROOM_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Document ID:** {doc_id}\n")
            f.write(f"**Source Type:** {source_type}\n")
            f.write("-" * 40 + "\n\n")
            f.write(content)
            
    print("\n✅ Document import complete! Next step: run 'python run_ingestion.py' to embed them.")

if __name__ == "__main__":
    import_documents()
