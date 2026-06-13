# Federated M&A Due Diligence Swarm

> Graph-of-Graphs multi-agent architecture for autonomous corporate data room auditing.
> **Modules A (Data Layer), B (Domain Agents), and C (Mesh Interface)** are implemented here.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    QUERY (from Agent)                          │
└────────────────────────┬───────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Stage 1            │
              │  Query Decomposer   │  ← LLM (Groq / Ollama)
              │  (LLM-powered)      │
              └──────────┬──────────┘
                         │  [{subquery, domain}, …]
          ┌──────────────▼──────────────┐
          │  Stage 2: Hybrid Search     │  parallel per sub-query
          │  ┌──────────┐ ┌──────────┐  │
          │  │ Dense    │ │  BM25    │  │  ← BGE-M3 + ChromaDB
          │  │ (cosine) │ │ Keyword  │  │  ← rank_bm25
          │  └────┬─────┘ └────┬─────┘  │
          └───────┼────────────┼─────────┘
                  └─────┬──────┘
              ┌──────────▼──────────┐
              │  Stage 3            │
              │  RRF Fusion         │  ← rank-free score merging
              └──────────┬──────────┘
              ┌──────────▼──────────┐
              │  Stage 4            │
              │  Cross-Encoder      │  ← BGE-reranker-v2-m3
              │  Reranking          │
              └──────────┬──────────┘
              ┌──────────▼──────────┐
              │  Stage 5            │
              │  Context Assembly   │  → injected into Agent prompt
              └─────────────────────┘
```

---

## Project Structure

```
ma_due_diligence/
├── .env.example              ← Copy to .env and fill in your keys
├── requirements.txt
├── run_ingestion.py          ← One-shot: load & index your data room
├── run_query.py              ← Interactive query loop for manual testing
│
├── module_a/                 ← Data Layer (this module)
│   ├── config.py             ← All tuneable parameters
│   ├── ingestion/
│   │   ├── loader.py         ← PDF / DOCX / TXT / MD loading + domain inference
│   │   └── chunker.py        ← Sentence-aware sliding-window chunker
│   ├── retrieval/
│   │   ├── embedder.py       ← BGE-M3 dense + sparse (singleton)
│   │   ├── vector_store.py   ← ChromaDB cosine search
│   │   ├── bm25_store.py     ← BM25Okapi keyword index (persisted)
│   │   ├── hybrid_search.py  ← Parallel search → RRF
│   │   └── rrf.py            ← Reciprocal Rank Fusion
│   ├── reranker.py           ← BGE-reranker-v2-m3 (singleton)
│   ├── query_decomposer.py   ← Stage 1 LLM decomposition
│   ├── pipeline.py           ← RAGPipeline — the main entry point
│   └── tests/
│       ├── conftest.py       ← Synthetic M&A fixtures (session-scoped)
│       └── test_pipeline.py  ← Unit + integration tests, poison pill test
│
├── module_b/                 ← Domain Agents (Workers)
│   ├── agent_factory.py      ← Factory pattern for agent instantiation
│   ├── base_agent.py         ← Base LLM invocation and shared logic
│   ├── config.py             ← Module B configuration parameters
│   ├── cyber_agent.py        ← Cybersecurity Domain Agent
│   ├── financial_agent.py    ← Financial Domain Agent
│   ├── hr_agent.py           ← HR Domain Agent
│   ├── legal_agent.py        ← Legal Domain Agent
│   ├── schemas.py            ← Output schemas for strict extractions
│   └── test_agents.py        ← Isolated extraction tests
│
├── module_c/                 ← Mesh Interface (P2P Debate)
│   ├── __init__.py
│   ├── agent_nodes.py        ← LangGraph node factories for domain agents
│   ├── debate_monitor.py     ← Kill-switch detection and mesh routing
│   ├── mesh_graph.py         ← Mesh sub-graph StateGraph assembly
│   ├── mesh_state.py         ← Typed state schema (inboxes, threads)
│   ├── peer_tools.py         ← P2P tool factories (send/resolve)
│   └── tests/
│       └── test_mesh.py      ← P2P routing and kill-switch tests
│
└── data/
    └── data_room/            ← Drop your M&A documents here
```

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY for dev mode
```

### 3. Add documents to the data room
Drop any `.pdf`, `.docx`, `.txt`, or `.md` files into `data/data_room/`.
Domain (legal / financial / hr / cybersecurity) is auto-inferred from filenames
and content — no manual tagging required.

### 4. Ingest
```bash
python run_ingestion.py
```
Ingestion is **idempotent** — re-running it on the same files is safe.

### 5. Query
```bash
# Interactive sample queries
python run_query.py

# Single custom query
python run_query.py "What are the change-of-control penalties and cybersecurity risks?"
```

### 6. Run tests
```bash
pytest module_a/tests/ -v
```

---

## Configuration Reference (`module_a/config.py`)

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `groq` | `groq` for dev, `ollama` for production |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID |
| `OLLAMA_MODEL` | `llama3:8b` | Local Ollama model |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` for GPU acceleration |
| `chunk_size` | `512` | Target words per chunk |
| `chunk_overlap` | `64` | Overlap words between chunks |
| `top_k_retrieval` | `20` | Candidates per search arm |
| `top_k_reranked` | `5` | Final chunks after reranking |
| `rrf_k` | `60` | RRF constant |
| `max_subqueries` | `4` | Max query decomposition branches |

---

## Switching to Production (Ollama)

```bash
# .env
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3:8b
EMBEDDING_DEVICE=cuda   # if GPU available
```

Make sure Ollama is running locally:
```bash
ollama serve
ollama pull llama3:8b
```

---

## Module Roadmap (spec §4)

| Module | Status | Description |
|---|---|---|
| **A** | ✅ Complete | 5-Stage RAG pipeline (this module) |
| **B** | ✅ Complete | Domain agents — Legal, Financial, HR, Cyber |
| **C** | ✅ Complete | Mesh interface — P2P tool-calling protocol |
| **D** | 🔜 Next | Leader Agent — LangGraph state + kill switch |

**Integration rule:** Do not run the full graph until each module passes its isolated unit tests.
