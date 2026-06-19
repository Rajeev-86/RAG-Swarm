# Federated M&A Due Diligence Swarm (RAG-Swarm)

An enterprise-grade, multi-agent AI system designed to automate and cross-validate Mergers & Acquisitions (M&A) due diligence. 

Built natively on **LangGraph**, the system orchestrates a swarm of domain-specific experts (Legal, Financial, HR, Cybersecurity) that retrieve data from a shared virtual data room, identify risks, and autonomously debate cross-domain implications in a secure peer-to-peer mesh network.

## 🏗️ Architecture

The system is strictly decoupled into four modular layers:

### Module A: Data Layer (5-Stage RAG)
A production-ready RAG pipeline ensuring agents receive highly relevant context.
* **Idempotent Ingestion:** Deterministic SHA-256 chunk hashing prevents index duplication on re-runs.
* **Hybrid Search:** Combines semantic search (ChromaDB/cosine similarity) with exact keyword matching (BM25Okapi).
* **Reciprocal Rank Fusion (RRF):** Merges dense and sparse retrieval signals natively without arbitrary normalisation.
* **Cross-Encoder Reranking:** Re-scores the fused candidate pool for maximum contextual precision.

### Module B: Domain Agents
Four specialised LangChain agents: `Financial`, `Legal`, `HR`, and `Cybersecurity`.
* Each agent operates independently during the initial extraction phase.
* Utilises strict Pydantic schemas (`AgentResult`, `PeerQuery`, `Finding`) to output structured risks and scores.
* Agents identify inter-domain blind spots and generate `PeerQueries` for cross-validation.

### Module C: Mesh Interface (P2P Network)
A decentralised communication graph where agents collaborate.
* **Tool Calling:** Agents utilise `send_peer_query` and `resolve_debate` tools via dynamically bound LangChain `StructuredTool` interfaces.
* **Debate Limits:** Enforces a strict 3-turn limit per thread to prevent infinite conversational loops.
* **Kill-Switch Intervention:** Automatically intercepts "context fragmentation" if an agent pair triggers multiple concurrent, unresolved threads.

### Module D: Leader Agent (The State Hub)
The central orchestration layer managing the swarm's lifecycle.
* **Deterministic Execution:** Uses LangGraph's explicit graph-based execution model (StateGraph) for deterministic coordination, conditional routing, and auditable trails.
* **State Persistence:** Implements a SQLite checkpointer to persist memory across sessions and ensure agents can resume interrupted workflows.
* **Forced Summaries:** LLM-powered nodes generate synthesised findings if the kill-switch interrupts a deadlocked debate.

---

## ⚙️ Prerequisites & Setup

The framework is model-agnostic, supporting both local execution and cloud APIs.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rajeev-86/rag-swarm.git](https://github.com/rajeev-86/rag-swarm.git)
   cd rag-swarm

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment:**
Create a `.env` file in the root directory and configure your chosen API keys:
```env
# Required for DeepEval / Swarm LLM-as-a-judge
GOOGLE_API_KEY="your-gemini-key" 

# Optional: Set the swarm backend (defaults to 'groq')
LLM_BACKEND="groq" # Options: groq, ollama
GROQ_API_KEY="your-groq-key"
GROQ_MODEL="llama3-70b-8192"

```



---

## 🚀 Usage Guide

### 1. Ingest the Data Room

Import the test dataset and populate the ChromaDB and BM25 databases.

```bash
python import_hf_dataset.py
python run_ingestion.py

```

### 2. Execute the Swarm

Run the master script to trigger the full A → B → Bridge → C/D pipeline. The swarm will retrieve evidence, extract risks, merge duplicate tasks, and dispatch debates to the LangGraph mesh.

```bash
python run_audit.py

```

---

## 📊 Evaluation (DeepEval)

The project utilises DeepEval (powered by Gemini) to quantitatively score the non-deterministic outputs of the system.

**Phase 1: Retrieval Metrics**
Evaluates Module A's 5-stage pipeline for Contextual Precision, Contextual Recall, and Faithfulness.

```bash
python evaluate_hf_questions.py

```

**Phase 2: Agent Trajectory Metrics**
Parses LangGraph execution logs to grade the swarm's Tool Correctness, Step Efficiency, and overall Plan Adherence.

```bash
python evaluate_swarm_metrics.py

```

---

*Note: This architecture avoids the "dumb zone" of context degradation by using isolated subagents with specific tools rather than a monolithic master agent.*