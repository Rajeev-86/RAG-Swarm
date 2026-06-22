"""
module_d/observability.py
─────────────────────────
Initialises observability and tracing for the LangGraph swarm.
Supports both LangSmith (Cloud) and Arize Phoenix (Local OTel).

Usage:
    from module_d.observability import setup_tracing
    setup_tracing()
"""
import os
import logging

logger = logging.getLogger(__name__)

def setup_tracing():
    backend = os.getenv("TRACING_BACKEND", "none").lower()
    
    if backend == "langsmith":
        # LangSmith requires no code instrumentation for LangChain/LangGraph,
        # it natively hooks in via environment variables once TRACING_V2 is active.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        project = os.getenv("LANGCHAIN_PROJECT", "RAG-Swarm-Audit")
        os.environ["LANGCHAIN_PROJECT"] = project
        print(f"✅ [Observability] LangSmith tracing enabled (Project: {project})")
        
    elif backend == "phoenix":
        try:
            from phoenix.otel import register
            from openinference.instrumentation.langchain import LangChainInstrumentor
            
            register(project_name=os.getenv("LANGCHAIN_PROJECT", "RAG-Swarm-Audit"))
            LangChainInstrumentor().instrument()
            
            print("✅ [Observability] Arize Phoenix tracing enabled.")
        except (ImportError, Exception):
            print("❌ [Observability] Failed to initialize Phoenix. Did you run `pip install arize-phoenix openinference-instrumentation-langchain`?")
            
    else:
        print("ℹ️ [Observability] Tracing disabled (TRACING_BACKEND not set to 'langsmith' or 'phoenix').")