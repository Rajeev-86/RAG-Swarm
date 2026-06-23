"""
module_a/query_decomposer.py
─────────────────────────────
Stage 1: Query Decomposition.

A complex M&A query like:
  "What are the cybersecurity risks and do any contracts require security audits?"

should become two sub-queries, each routed to the right search domain:
  1. { subquery: "unpatched vulnerabilities and security audit findings",    domain: "cybersecurity" }
  2. { subquery: "contractual obligations requiring security audit or SOC 2", domain: "legal" }

The decomposer calls the configured LLM (Groq in dev, Ollama in production)
and parses its JSON response. On any parse failure it gracefully falls back
to a single passthrough sub-query so the pipeline never hard-fails.

Backend switching:
  Controlled by cfg.llm_backend (set in .env).
  Groq: ~200 ms latency, ideal for dev/CI.
  Ollama: runs fully locally, suitable for production air-gap environments.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
  
from module_a.config import cfg, LLMBackend


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a senior M&A due diligence analyst. "
    "Your job is to decompose complex queries into focused domain-specific sub-queries."
)

_USER_TEMPLATE = """Break the following M&A due diligence query into at most {max_n} focused sub-queries.
Each sub-query MUST target exactly one domain: legal, financial, hr, or cybersecurity.

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Each element: {{"subquery": "<focused question>", "domain": "<domain>"}}

Query: {query}"""


# ── LLM Callers ───────────────────────────────────────────────────────────────

def _call_groq(user_prompt: str) -> str:
    from groq import Groq
    if not cfg.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    client   = Groq(api_key=cfg.groq_api_key)
    response = client.chat.completions.create(
        model    = cfg.groq_model,
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = 0.0,   # Deterministic decomposition
        max_tokens  = 512,
    )
    return response.choices[0].message.content.strip()


def _call_ollama(user_prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model   = cfg.ollama_model,
        messages= [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options = {"temperature": 0.0},
    )
    return response["message"]["content"].strip()


def _call_llm(prompt: str) -> str:
    if cfg.llm_backend == LLMBackend.GROQ:
        return _call_groq(prompt)
    return _call_ollama(prompt)


# ── Public API ────────────────────────────────────────────────────────────────

def decompose_query(query: str) -> list[dict]:
    """
    Decompose a complex M&A query into domain-routed sub-queries.

    Returns:
        List of dicts: [{"subquery": str, "domain": str}, …]
        Always returns at least one item (fallback = original query, domain = "unknown").
    """
    prompt = _USER_TEMPLATE.format(query=query, max_n=cfg.max_subqueries)

    try:
        raw   = _call_llm(prompt)
        # Strip any accidental markdown fences the LLM might emit
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)

        valid = [
            s for s in parsed
            if isinstance(s, dict)
            and "subquery" in s
            and "domain" in s
            and s["domain"] in cfg.domains + ["unknown"]
        ]
        if valid:
            print(f"[QueryDecomposer] ✓ Decomposed into {len(valid)} sub-queries.")
            return valid

        print("[QueryDecomposer] ⚠ No valid sub-queries parsed — using fallback.")

    except Exception as exc:
        print(f"[QueryDecomposer] ⚠ LLM call/parse failed ({exc}) — using fallback.")

    # Graceful fallback: treat the whole query as a single unknown-domain sub-query
    return [{"subquery": query, "domain": "unknown"}]