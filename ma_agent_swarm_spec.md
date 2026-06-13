# System Architecture & Implementation Specification
## Federated Multi-Agent M&A Due Diligence Swarm (Mesh Topology)

**Context for AI Agent Reader:** You are reading the technical specification for a multi-agent AI system. When assisting the human developer with generating code, debugging, or designing components for this project, you MUST refer to the specifications laid out in this document as the absolute source of truth.

---

## 1. Project Overview & Problem Statement
The objective is to build a decentralized swarm of highly specialized AI agents (Legal, Financial, HR, Cybersecurity) that autonomously audit a massive corporate data room containing thousands of heterogeneous documents. 

**The Core Bottleneck Addressed:** Different domains require drastically different retrieval and reasoning strategies. Traditional rigid RAG pipelines fail in enterprise scenarios. However, uncoordinated parallel agent systems suffer from severe error amplification (up to 17.2x) when hallucinated context is shared peer-to-peer.
**The Solution:** A hybrid Graph-of-Graphs architecture combining P2P Mesh communication with a centralized Leader Agent for stateful checkpointing and interruption.

---

## 2. The Technology Stack (100% Free & Open-Source)
The stack is specifically chosen to run locally on available hardware to avoid cloud API costs while maintaining 2026 production-grade capabilities.

* **Orchestration / Multi-Agent Framework:** * **LangGraph:** Handles the Leader Agent, global cyclical loops, state management, and checkpointing.
    * **AutoGen (AG2):** Can be optionally utilized for the mesh/debate sub-layer due to its native P2P capabilities.
* **LLM Inference (The Brains):** **Ollama** running quantized models locally (e.g., `Llama-3 8B` or `Mistral`). Groq API can be utilized for fast inference during component testing.
* **Vector Database:** **ChromaDB** (for rapid Python iteration) or **pgvector** (for robust production hybrid search).
* **Embeddings (Sparse & Dense):** `BAAI/bge-m3` via HuggingFace (critical for generating both dense embeddings and sparse BM25 vectors simultaneously).
* **Cross-Encoder / Reranker:** `BAAI/bge-reranker-v2-m3` running locally.
* **Evaluation & Observability:** **DeepEval** (for computing RAG and Agentic metrics) and **Arize Phoenix** / **LangSmith** (for execution tracing).

---

## 3. The Architecture Blueprint

### 3.1. The 5-Stage RAG Engine (Data Layer)
Implements the 2026 production standard for retrieval:
1.  **Query Decomposition:** Break down complex, multi-hop queries into domain-specific sub-queries.
2.  **Hybrid Search:** Execute parallel searches combining BM25 keyword matching with dense semantic embeddings.
3.  **Reciprocal Rank Fusion (RRF):** Merge and normalize the rankings from the dense and sparse searches.
4.  **Cross-Encoder Reranking:** Pass the top-K results through `bge-reranker-v2-m3` for rigorous precision filtering.
5.  **Context Injection:** Feed the verified, reranked chunks directly into the specific Agent's context window.

### 3.2. Graph-of-Graphs Orchestration
* **The Global Loop (Leader Agent):** Maintained via LangGraph. Holds the `SharedMemory`, `Data_Room_Status`, and `Audit_Log`.
* **The Sub-Graph (Mesh Network):** Domain agents communicate P2P. If the Financial Agent flags a revenue discrepancy, it directly queries the Legal Agent to fetch related penalty clauses without routing through the Leader.
* **The Kill Switch (Crucial):** The Leader passively monitors the LangGraph state. If a P2P debate exceeds turn limits (e.g., >3 turns) or retrieved context becomes highly fragmented, the Leader interrupts, forces a summary, saves the state to disk (Checkpoint), and resets the sub-agents' context windows.

---

## 4. Phased Development Modules

Development MUST occur in strict isolation before integration. Do not attempt to run the entire graph concurrently until individual modules pass unit tests.

* **Module A: Data Layer (5-Stage RAG):** Build the ingestion, semantic chunking, and hybrid search pipeline. *Test: Multi-hop retrieval accuracy.*
* **Module B: Domain Agents (Workers):** Define strict system prompts for Legal, Financial, HR, and Cyber agents. *Test: Isolated extraction capabilities.*
* **Module C: Mesh Interface (Debate):** Develop the P2P tool-calling protocol allowing agents to pass contextual arguments to each other.
* **Module D: Leader Agent (State Hub):** Implement LangGraph state dict handling, disk checkpointing, and the intervention "kill switch."

---

## 5. Evaluation Protocol

You must evaluate deterministic data retrieval (RAG) completely independently from non-deterministic reasoning trajectories (Agents).

### 5.1. RAG Metrics (DeepEval)
* **Context Precision:** Did the Cross-Encoder rank the exact necessary clause at rank #1?
* **Context Recall:** Did the hybrid search retrieve the clause at all?
* **Faithfulness:** Does the agent's extracted claim match the source chunk exactly (Zero-fabrication validation)?

### 5.2. Swarm Metrics (Agentic Trajectories via LLM-as-a-Judge)
* **Tool Correctness:** Are arguments formatted and passed correctly between agents during P2P handoffs?
* **Step Efficiency:** Are adversarial disputes resolved under the defined turn limit?
* **Plan Adherence:** Did the Leader Agent successfully detect a loop and trigger a checkpoint?

### 5.3. Stress Testing: The "Poison Pill" Test
To test the mitigation of the 17.2x error amplification risk:
1. Explicitly inject a contradictory, synthetic document into the data room.
2. Prompt the Financial Agent to evaluate it and debate the Legal Agent.
3. *Success Criteria:* The Leader Agent must intercept the resulting hallucination cascade and force a state reset before the Legal Agent permanently corrupts its own context window.

---
## References
- Anthropic, 2026. "How we built our multi-agent research system" | https://www.anthropic.com/engineering/multi-agent-research-system
- Appycodes, 2026. "Building a Production RAG Pipeline: Chunking, Embeddings, Retrieval, Caching" | https://appycodes.dev/blog/production-rag-pipeline-2026/
- Atlan, 2026. "RAGAS, TruLens, DeepEval: LLM Evaluation Frameworks" | https://atlan.com/know/llm-evaluation-frameworks-compared/
- Braintrust, 2026. "Best RAG Evaluation Tools in 2026, Compared" | https://www.braintrust.dev/articles/best-rag-evaluation-tools
- Firecrawl, 2026. "The best open source frameworks for building AI agents in 2026" | https://www.firecrawl.dev/blog/best-open-source-agent-frameworks
- Maxim AI, 2026. "Top 5 RAG Evaluation Platforms in 2026" | https://www.getmaxim.ai/articles/top-5-rag-evaluation-platforms-in-2026/
- Metafied Lab, 2026. "How to Build a Production-Ready RAG Pipeline in 2026" | https://metafiedlab.com/blog/how-to-build-a-production-ready-rag-pipeline-in-2026/
- MDPI, 2026. "Multi-Agent Coordination Strategies vs. Retrieval-Augmented Generation in LLMs" | https://www.mdpi.com/2079-9292/14/24/4883
- YAITEC, 2026. "The 7 best free AI agent frameworks for developers in 2026" | https://www.yaitec.com/en/blog/best-frameworks-agentes-ai-gratuitos-2026
