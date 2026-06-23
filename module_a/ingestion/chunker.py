"""
module_a/ingestion/chunker.py
──────────────────────────────
Sentence-aware sliding-window chunker.

Strategy:
  1. Split the document text into sentences.
  2. Accumulate sentences into a chunk until the word count hits `chunk_size`.
  3. Start the next chunk with the last `chunk_overlap` words so context bleeds
     across chunk boundaries — critical for legal clause detection that often
     spans multiple sentences.

Word counts (not character counts) are used because BGE-M3 tokenises on
sub-word units; 512 words ≈ 680–750 tokens, comfortably within the 8 192-token
context window.
"""

import re
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from module_a.ingestion.loader import RawDocument
from module_a.config import cfg


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    chunk_id:     str
    content:      str
    source:       str
    domain:       str
    chunk_index:  int
    total_chunks: int
    metadata:     dict = field(default_factory=dict)


# ── Sentence Splitter ─────────────────────────────────────────────────────────

# Handles common legal abbreviations (e.g. "Sec.", "Art.", "U.S.") by requiring
# the token after the period to start with a capital letter followed by more chars.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z][a-z])')

def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


# ── Core Chunking Logic ───────────────────────────────────────────────────────

def chunk_text(
    text:       str,
    chunk_size: int = None,
    overlap:    int = None,
) -> list[str]:
    """
    Chunk a plain-text string into overlapping word-count windows that
    respect sentence boundaries.

    Returns:
        A list of non-empty string chunks.
    """
    chunk_size = chunk_size or cfg.chunk_size
    overlap    = overlap if overlap is not None else cfg.chunk_overlap

    sentences = _split_sentences(text)
    chunks:      list[str]  = []
    window:      list[str]  = []   # accumulates words for current chunk
    word_count:  int        = 0

    for sentence in sentences:
        words = sentence.split()
        n     = len(words)

        # If adding this sentence would overflow the window, flush first
        if word_count + n > chunk_size and window:
            chunks.append(" ".join(window))
            # Seed the next window with the overlap tail
            window     = window[-overlap:] if overlap > 0 else []
            word_count = len(window)

        window.extend(words)
        word_count += n

    # Flush the final partial chunk
    if window:
        chunks.append(" ".join(window))

    return [c for c in chunks if c.strip()]


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    """Chunk a single RawDocument into DocumentChunk objects."""
    raw_chunks = chunk_text(doc.content)
    total      = len(raw_chunks)
    chunks: list[DocumentChunk] = []

    for i, content in enumerate(raw_chunks):
        # Generate a deterministic ID based on source, index, and text content
        hash_input = f"{doc.source}::{i}::{content}".encode("utf-8")
        deterministic_id = hashlib.sha256(hash_input).hexdigest()

        chunks.append(DocumentChunk(
            chunk_id    = deterministic_id,
            content     = content,
            source      = doc.source,
            domain      = doc.domain,
            chunk_index = i,
            total_chunks= total,
            metadata    = {
                **doc.metadata,
                "chunk_index":  i,
                "total_chunks": total,
            },
        ))
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[DocumentChunk]:
    """Chunk all documents and return a flat list of DocumentChunk objects."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    print(
        f"[Chunker] ✓ Produced {len(all_chunks)} chunks "
        f"from {len(docs)} documents"
    )
    return all_chunks