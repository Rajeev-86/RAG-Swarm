"""
module_b/agent_factory.py

Factory and convenience runners for Module B domain agents.

Usage (single agent):
    from module_b.agent_factory import create_agent
    from module_b.schemas import DomainType

    agent = create_agent(DomainType.LEGAL)
    result = agent.analyze(retrieval_result)

Usage (all agents in parallel — primary integration path):
    from module_b.agent_factory import run_all_agents

    all_results = run_all_agents(retrieval_result)
    # Returns: dict[DomainType, AgentResult]

Usage (consolidated report):
    from module_b.agent_factory import build_consolidated_report

    report = build_consolidated_report(all_results)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from module_b.agents.cyber_agent import CyberAgent
from module_b.agents.financial_agent import FinancialAgent
from module_b.agents.hr_agent import HRAgent
from module_b.agents.legal_agent import LegalAgent
from module_b.base_agent import BaseAgent
from module_b.config import ModuleBConfig
from module_b.schemas import AgentResult, DomainType, RiskLevel

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

_AGENT_REGISTRY: dict[DomainType, type[BaseAgent]] = {
    DomainType.LEGAL: LegalAgent,
    DomainType.FINANCIAL: FinancialAgent,
    DomainType.HR: HRAgent,
    DomainType.CYBERSECURITY: CyberAgent,
}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_agent(
    domain: DomainType,
    config: Optional[ModuleBConfig] = None,
) -> BaseAgent:
    """
    Instantiate a single domain agent by DomainType.

    Args:
        domain:  One of DomainType.LEGAL / FINANCIAL / HR / CYBERSECURITY
        config:  Optional ModuleBConfig; reads .env if None

    Returns:
        A fully initialised domain agent ready to call .analyze()
    """
    agent_cls = _AGENT_REGISTRY.get(domain)
    if agent_cls is None:
        raise ValueError(
            f"Unknown domain: {domain!r}. "
            f"Valid values: {[d.value for d in _AGENT_REGISTRY]}"
        )
    return agent_cls(config=config)


# ── Parallel runner ───────────────────────────────────────────────────────────

def run_all_agents(
    retrieval_result,
    config: Optional[ModuleBConfig] = None,
    max_workers: int = 4,
) -> dict[DomainType, AgentResult]:
    """
    Run all four domain agents in parallel threads against the same RetrievalResult.

    Each agent operates independently on its own domain slice (domain_breakdown).
    No shared mutable state — safe for parallel execution.

    Args:
        retrieval_result: Module A RAGPipeline.retrieve() output (RetrievalResult)
        config:           Shared ModuleBConfig for all agents; reads .env if None
        max_workers:      Thread pool size (default 4 — one per agent)

    Returns:
        dict mapping DomainType → AgentResult for each domain
    """
    cfg = config or ModuleBConfig.from_env()
    cfg.validate()

    agents = {domain: agent_cls(config=cfg) for domain, agent_cls in _AGENT_REGISTRY.items()}
    results: dict[DomainType, AgentResult] = {}

    logger.info(f"[factory] Launching {len(agents)} domain agents in parallel...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(agent.analyze, retrieval_result): domain
            for domain, agent in agents.items()
        }
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
                results[domain] = result
                logger.info(
                    f"[factory] ✓ {domain.value} agent completed | "
                    f"findings={len(result.findings)} | "
                    f"risk_score={result.total_risk_score:.1f} | "
                    f"escalation={result.requires_escalation}"
                )
            except Exception as exc:
                logger.error(f"[factory] ✗ {domain.value} agent failed: {exc}")
                # Return empty result so the swarm is never partially broken
                results[domain] = AgentResult(
                    agent_id=f"{domain.value}_agent",
                    domain=domain,
                    query=getattr(retrieval_result, "query", "unknown"),
                    findings=[],
                    summary=f"Agent failed with exception: {exc}",
                    total_risk_score=0.0,
                    requires_escalation=True,
                    escalation_reason=f"Agent runtime failure: {exc}",
                )

    return results


# ── Consolidated report builder ───────────────────────────────────────────────

def build_consolidated_report(
    results: dict[DomainType, AgentResult],
) -> dict:
    """
    Aggregates all four AgentResults into a single consolidated audit report dict.
    This is the payload Module D's Leader Agent will checkpoint to disk.

    Structure:
        {
            "overall_risk_score": float,
            "requires_escalation": bool,
            "escalation_reasons": [str, ...],
            "total_findings": int,
            "risk_breakdown": {"CRITICAL": N, "HIGH": N, ...},
            "pending_peer_queries": [{...}, ...],
            "domain_results": {"legal": {...}, "financial": {...}, ...},
        }
    """
    all_findings = [f for r in results.values() for f in r.findings]
    all_peer_queries = [pq for r in results.values() for pq in r.peer_queries]
    escalation_reasons = [
        r.escalation_reason
        for r in results.values()
        if r.requires_escalation and r.escalation_reason
    ]

    # Weighted average risk score across all domains
    scores = [r.total_risk_score for r in results.values() if r.total_risk_score > 0]
    overall_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    risk_breakdown: dict[str, int] = {r.value: 0 for r in RiskLevel}
    for finding in all_findings:
        risk_breakdown[finding.risk_level.value] += 1

    return {
        "overall_risk_score": overall_score,
        "requires_escalation": any(r.requires_escalation for r in results.values()),
        "escalation_reasons": escalation_reasons,
        "total_findings": len(all_findings),
        "risk_breakdown": risk_breakdown,
        "pending_peer_queries": [pq.model_dump() for pq in all_peer_queries],
        "domain_results": {
            domain.value: result.model_dump()
            for domain, result in results.items()
        },
    }
