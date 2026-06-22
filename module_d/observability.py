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
            import phoenix as px
            from openinference.instrumentation.langchain import LangChainInstrumentor
            
            # Launch the local Phoenix UI server
            session = px.launch_app()
            
            # Hook OpenTelemetry into LangChain
            LangChainInstrumentor().instrument()
            
            print(f"✅ [Observability] Arize Phoenix tracing enabled!")
            print(f"🔍 Phoenix UI running locally at: {session.url}")
        except ImportError:
            print("❌ [Observability] Failed to initialize Phoenix. Did you run `pip install arize-phoenix openinference-instrumentation-langchain`?")
            
    else:
        print("ℹ️ [Observability] Tracing disabled (TRACING_BACKEND not set to 'langsmith' or 'phoenix').")