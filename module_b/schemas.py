"""
module_b/schemas.py

All Pydantic models for Module B I/O.

Forward-compatibility contract:
  • PeerQuery   → consumed by Module C (Mesh Interface) to route P2P agent calls
  • AgentResult → consumed by Module D (Leader Agent) for state management & checkpointing
  
Do not change field names without updating the corresponding Module C/D stubs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def numeric(self) -> float:
        """Numeric weight used for total_risk_score computation."""
        return {"LOW": 2.0, "MEDIUM": 4.5, "HIGH": 7.0, "CRITICAL": 9.5}[self.value]


class DomainType(str, Enum):
    LEGAL = "legal"
    FINANCIAL = "financial"
    HR = "hr"
    CYBERSECURITY = "cybersecurity"


# ── Module C handshake ─────────────────────────────────────────────────────────

class PeerQuery(BaseModel):
    """
    A structured request dispatched from one domain agent to another.
    Module C's Mesh Interface reads this to route P2P tool calls.

    Lifecycle:
        Agent B produces PeerQuery → Module C routes it → target Agent answers →
        answer is injected back into the source Agent's context for re-analysis.
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_domain: DomainType
    target_domain: DomainType
    question: str                     # The precise question for the target agent
    context_snippet: str              # The evidence chunk that prompted this query
    urgency: RiskLevel                # HIGH/CRITICAL queries get priority routing in Module C
    resolved: bool = False            # Module C sets this to True after answer is received
    answer: Optional[str] = None      # Module C writes the target agent's response here


# ── Core finding ───────────────────────────────────────────────────────────────

class Finding(BaseModel):
    """
    A single extracted risk or observation. The atomic unit of the audit trail.
    Every Finding is fully traceable back to its source RAG chunk(s).
    """
    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: DomainType
    title: str
    description: str
    risk_level: RiskLevel

    # Traceability — chunk IDs from Module A's RetrievalResult
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_quote: str = ""          # Verbatim excerpt from the source document

    flags: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # ── Module C handshake fields ──────────────────────────────────────────
    requires_peer_review: bool = False
    peer_review_target: Optional[DomainType] = None
    peer_review_question: Optional[str] = None


# ── Agent result (top-level payload) ──────────────────────────────────────────

class AgentResult(BaseModel):
    """
    The complete output of a single domain agent's analysis run.
    This is the primary payload passed to Module C and Module D.

    Module D (Leader Agent) reads:
        - requires_escalation  → trigger checkpoint / kill-switch
        - peer_queries         → pass to Module C's mesh router

    Module C (Mesh Interface) reads:
        - peer_queries         → dispatch P2P calls to target agents
    """
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str
    domain: DomainType
    query: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""
    total_risk_score: float = Field(default=0.0, ge=0.0, le=10.0)

    # ── Module D handshake fields ──────────────────────────────────────────
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None

    # ── Module C handshake fields ──────────────────────────────────────────
    # Compiled automatically from findings that set requires_peer_review=True
    peer_queries: list[PeerQuery] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.risk_level == RiskLevel.CRITICAL]

    @property
    def high_risk_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]

    @property
    def risk_summary(self) -> dict[str, int]:
        """Count of findings per risk level."""
        counts: dict[str, int] = {r.value: 0 for r in RiskLevel}
        for f in self.findings:
            counts[f.risk_level.value] += 1
        return counts