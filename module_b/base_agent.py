"""
module_b/base_agent.py

Abstract base class for all four domain agents.

Responsibilities:
  1. Extract domain-relevant chunks from a Module A RetrievalResult
  2. Format those chunks as numbered XML context blocks for the LLM
  3. Build the user-turn extraction prompt (shared logic + domain hints)
  4. Call the Groq API with JSON mode enforced
  5. Parse and validate the structured JSON response into AgentResult
  6. Compile PeerQuery objects from findings that flagged requires_peer_review
  7. Auto-trigger escalation when total_risk_score >= threshold or CRITICAL found

Subclasses define:
  - DOMAIN          (DomainType)
  - SYSTEM_PROMPT   (str)
  - EXTRACTION_HINTS (str, optional)  — appended to the user prompt
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from groq import Groq

from module_b.config import ModuleBConfig
from module_b.schemas import AgentResult, DomainType, Finding, PeerQuery, RiskLevel

logger = logging.getLogger(__name__)


class BaseAgent(ABC):

    # ── Subclass-defined class attributes ──────────────────────────────────
    DOMAIN: DomainType
    SYSTEM_PROMPT: str
    EXTRACTION_HINTS: str = ""  # Domain-specific focus notes appended to the user prompt

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, config: ModuleBConfig | None = None) -> None:
        self.config = config or ModuleBConfig.from_env()
        self._client = Groq(api_key=self.config.groq_api_key)
        self.agent_id = f"{self.DOMAIN.value}_agent"
        logger.debug(f"[{self.agent_id}] Initialized with model={self.config.groq_model}")

    # ── Public interface ─────────────────────────────────────────────────────

    def analyze(self, retrieval_result: Any, query_override: str | None = None) -> AgentResult:
        """
        Full analysis pipeline for one domain agent:

            RetrievalResult
                → extract domain chunks
                → format context
                → build user prompt
                → Groq JSON call
                → parse → AgentResult
                → compile PeerQuery list
                → auto-escalate if needed
        """
        query = query_override or retrieval_result.query
        context = self._extract_domain_context(retrieval_result)

        if not context.strip():
            logger.warning(
                f"[{self.agent_id}] No domain-relevant chunks found. "
                f"Returning empty result for query: {query!r}"
            )
            return self._empty_result(query, reason="No domain-relevant context was retrieved.")

        prompt = self._build_extraction_prompt(context=context, query=query)
        raw_response = self._call_llm(prompt)
        result = self._parse_llm_response(raw_response=raw_response, query=query)

        # ── Post-processing ────────────────────────────────────────────────
        result.peer_queries = self._compile_peer_queries(result.findings)
        result = self._apply_escalation_policy(result)
        result = self._domain_sanity_check(result)

        return result

    # ── Context extraction from Module A RetrievalResult ────────────────────

    def _extract_domain_context(self, retrieval_result: Any) -> str:
        """
        Pulls domain-specific chunks from the Module A RetrievalResult.

        Priority order:
          1. result.domain_breakdown[self.DOMAIN.value]   — pre-sliced by Module A (O(1))
          2. Filter result.retrieved_chunks by metadata.domain
          3. Fallback: all retrieved_chunks (cross-domain analysis)
        """
        chunks: list[dict] = []

        # ① Domain breakdown (fastest path — Module A already did the split)
        if hasattr(retrieval_result, "domain_breakdown") and retrieval_result.domain_breakdown:
            chunks = retrieval_result.domain_breakdown.get(self.DOMAIN.value, [])

        # ② Metadata filter over all chunks
        if not chunks and hasattr(retrieval_result, "retrieved_chunks"):
            chunks = [
                c for c in retrieval_result.retrieved_chunks
                if c.get("metadata", {}).get("domain", "") == self.DOMAIN.value
            ]

        # ③ Last resort: use all chunks (may help for cross-domain queries)
        if not chunks and hasattr(retrieval_result, "retrieved_chunks"):
            logger.debug(
                f"[{self.agent_id}] Domain filter empty — falling back to all chunks."
            )
            chunks = retrieval_result.retrieved_chunks

        return self._format_chunks(chunks)

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Formats a list of chunk dicts into numbered XML blocks for LLM context."""
        if not chunks:
            return ""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("document", chunk.get("content", chunk.get("text", ""))).strip()
            source = chunk.get("metadata", {}).get("source", "unknown_source")
            chunk_id = chunk.get("id", f"chunk_{i}")
            parts.append(
                f"<chunk id='{chunk_id}' source='{source}' index='{i}'>\n"
                f"{content}\n"
                f"</chunk>"
            )
        return "\n\n".join(parts)

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_extraction_prompt(self, context: str, query: str) -> str:
        """
        Builds the user-turn prompt. Concrete — subclasses extend via EXTRACTION_HINTS.
        The JSON schema is repeated here (not just in the system prompt) to reduce
        format-deviation on smaller Groq models.
        """
        hints_block = (
            f"\n\nDOMAIN-SPECIFIC FOCUS FOR THIS ANALYSIS:\n{self.EXTRACTION_HINTS}"
            if self.EXTRACTION_HINTS
            else ""
        )

        return f"""INVESTIGATION QUERY:
{query}
{hints_block}

DATA ROOM EXCERPTS (use chunk id values in source_chunk_ids):
{context}

---
Analyze the excerpts above. Return ONLY valid JSON matching this exact schema:

{{
  "findings": [
    {{
      "title": "<concise risk label, max 80 chars>",
      "description": "<detailed explanation of risk and deal impact>",
      "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
      "evidence_quote": "<verbatim excerpt from a chunk, max 400 chars>",
      "source_chunk_ids": ["<chunk id string>"],
      "flags": ["<short tag>"],
      "recommendations": ["<specific actionable step>"],
      "requires_peer_review": <true|false>,
      "peer_review_target": "<legal|financial|hr|cybersecurity|null>",
      "peer_review_question": "<question for the peer agent, or null>"
    }}
  ],
  "summary": "<2-3 sentence executive summary of this domain's overall risk posture>",
  "total_risk_score": <float 0.0-10.0>,
  "requires_escalation": <true|false>,
  "escalation_reason": "<reason string or null>"
}}"""

    # ── Groq API call ────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        """Calls Groq with JSON mode enforced. Returns raw JSON string."""
        logger.info(
            f"[{self.agent_id}] → Groq [{self.config.groq_model}] "
            f"| temp={self.config.temperature} | max_tokens={self.config.max_tokens}"
        )
        response = self._client.chat.completions.create(
            model=self.config.groq_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
        logger.debug(f"[{self.agent_id}] Raw response snippet: {raw[:200]!r}")
        return raw

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_llm_response(self, raw_response: str, query: str) -> AgentResult:
        """
        Parses the raw JSON string returned by Groq into a validated AgentResult.
        Handles minor LLM formatting deviations gracefully.
        """
        try:
            data = self._safe_parse_json(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"[{self.agent_id}] JSON parse failure: {exc}")
            return self._empty_result(
                query, reason=f"LLM returned malformed JSON: {exc}"
            )

        findings: list[Finding] = []
        for f_data in data.get("findings", []):
            try:
                peer_target_raw = f_data.get("peer_review_target")
                peer_target = DomainType(peer_target_raw) if peer_target_raw else None

                findings.append(Finding(
                    domain=self.DOMAIN,
                    title=f_data.get("title", "Untitled finding"),
                    description=f_data.get("description", ""),
                    risk_level=self._map_risk_level(f_data.get("risk_level", "MEDIUM")),
                    evidence_quote=f_data.get("evidence_quote", ""),
                    source_chunk_ids=f_data.get("source_chunk_ids", []),
                    flags=f_data.get("flags", []),
                    recommendations=f_data.get("recommendations", []),
                    requires_peer_review=bool(f_data.get("requires_peer_review", False)),
                    peer_review_target=peer_target,
                    peer_review_question=f_data.get("peer_review_question"),
                ))
            except Exception as exc:
                logger.warning(f"[{self.agent_id}] Skipping malformed finding: {exc}")
                continue

        # Compute risk score: use LLM value if sane, else compute from findings
        llm_score = float(data.get("total_risk_score", 0.0))
        computed_score = self._compute_risk_score(findings)
        total_risk_score = llm_score if 0.0 <= llm_score <= 10.0 else computed_score

        return AgentResult(
            agent_id=self.agent_id,
            domain=self.DOMAIN,
            query=query,
            findings=findings,
            summary=data.get("summary", ""),
            total_risk_score=round(total_risk_score, 2),
            requires_escalation=bool(data.get("requires_escalation", False)),
            escalation_reason=data.get("escalation_reason"),
        )

    # ── Peer query compilation ───────────────────────────────────────────────

    def _compile_peer_queries(self, findings: list[Finding]) -> list[PeerQuery]:
        """
        Converts per-finding peer_review flags into structured PeerQuery objects.
        These are the outbound P2P requests consumed by Module C's Mesh Interface.
        """
        queries: list[PeerQuery] = []
        for finding in findings:
            if finding.requires_peer_review and finding.peer_review_target:
                queries.append(PeerQuery(
                    source_domain=self.DOMAIN,
                    target_domain=finding.peer_review_target,
                    question=(
                        finding.peer_review_question
                        or f"Please cross-validate this finding: {finding.title}"
                    ),
                    context_snippet=finding.evidence_quote[:500],
                    urgency=finding.risk_level,
                ))
        return queries

    # ── Escalation policy ────────────────────────────────────────────────────

    def _apply_escalation_policy(self, result: AgentResult) -> AgentResult:
        """
        Auto-triggers escalation independent of what the LLM said if:
          - Any CRITICAL finding is present, OR
          - total_risk_score >= config.auto_escalation_score
        This is a safety net against the LLM under-reporting severity.
        """
        if result.critical_findings and not result.requires_escalation:
            result.requires_escalation = True
            result.escalation_reason = (
                f"[AUTO] {len(result.critical_findings)} CRITICAL finding(s) detected "
                f"by {self.agent_id}."
            )
        elif (
            result.total_risk_score >= self.config.auto_escalation_score
            and not result.requires_escalation
        ):
            result.requires_escalation = True
            result.escalation_reason = (
                f"[AUTO] Risk score {result.total_risk_score:.1f} exceeds threshold "
                f"{self.config.auto_escalation_score}."
            )
        return result

    # ── Utilities ────────────────────────────────────────────────────────────

    def _empty_result(self, query: str, reason: str = "") -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            domain=self.DOMAIN,
            query=query,
            findings=[],
            summary=reason or "No findings. Domain-relevant context was not available.",
            total_risk_score=0.0,
        )

    def _safe_parse_json(self, raw: str) -> dict:
        """Strips markdown fences and parses JSON."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return json.loads(cleaned.strip())

    def _map_risk_level(self, raw: str) -> RiskLevel:
        """Normalises arbitrary LLM risk strings to RiskLevel enum."""
        upper = raw.upper().strip()
        try:
            return RiskLevel(upper)
        except ValueError:
            synonyms = {
                "MINOR": RiskLevel.LOW,
                "MODERATE": RiskLevel.MEDIUM,
                "MAJOR": RiskLevel.HIGH,
                "SIGNIFICANT": RiskLevel.HIGH,
                "SEVERE": RiskLevel.CRITICAL,
                "EXTREME": RiskLevel.CRITICAL,
            }
            return synonyms.get(upper, RiskLevel.MEDIUM)

    def _compute_risk_score(self, findings: list[Finding]) -> float:
        """Weighted average of finding risk levels. Fallback when LLM gives bad score."""
        if not findings:
            return 0.0
        return round(
            sum(f.risk_level.numeric for f in findings) / len(findings), 2
        )

    # ── Abstract — subclasses can override for domain-specific behaviour ─────

    @abstractmethod
    def _domain_sanity_check(self, result: AgentResult) -> AgentResult:
        """
        Optional domain-specific post-processing hook.
        Override to add domain-specific validation logic (e.g., Legal agent
        checking that every CoC finding has a recommendation).
        Default implementation returns the result unchanged.
        """
        ...