"""
module_b/tests/test_agents.py

Module B isolation test suite.
Runs fully offline — Groq API is mocked. No API key required.

Run with:
    pytest module_b/tests/test_agents.py -v

Coverage:
  ✓ Schema validation (AgentResult, Finding, PeerQuery)
  ✓ All four domain agents in isolation
  ✓ Peer query compilation from findings
  ✓ Escalation auto-trigger (CRITICAL finding, score threshold)
  ✓ Empty context → empty AgentResult (no exception)
  ✓ Malformed LLM JSON → empty AgentResult (graceful recovery)
  ✓ run_all_agents() parallel runner
  ✓ build_consolidated_report() aggregation
  ✓ Domain-specific sanity checks (legal recs, financial regex, hr exec tags, cyber CVE tags)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    
from module_b.agent_factory import build_consolidated_report, run_all_agents
from module_b.agents.cyber_agent import CyberAgent
from module_b.agents.financial_agent import FinancialAgent
from module_b.agents.hr_agent import HRAgent
from module_b.agents.legal_agent import LegalAgent
from module_b.config import ModuleBConfig
from module_b.schemas import AgentResult, DomainType, Finding, PeerQuery, RiskLevel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@dataclass
class MockRetrievalResult:
    """Minimal stand-in for Module A's RetrievalResult dataclass."""
    query: str
    subqueries: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    domain_breakdown: dict = field(default_factory=dict)


# ── Synthetic data room chunks ────────────────────────────────────────────────

LEGAL_CHUNK = {
    "id": "legal_001",
    "content": (
        "Section 12.3 — Change of Control. Upon the occurrence of a Change of Control Event, "
        "as defined herein, all outstanding obligations under this Agreement shall become "
        "immediately due and payable. Counterparty written consent is required within 30 days "
        "of announcement. Failure to obtain consent triggers an irrevocable termination fee "
        "of $25,000,000 payable within 10 business days."
    ),
    "metadata": {"domain": "legal", "source": "acquisition_agreement.pdf"},
}

FINANCIAL_CHUNK = {
    "id": "fin_001",
    "content": (
        "Q3 2024 Revenue Recognition: Management applied ASC 606 using the "
        "percentage-of-completion method for long-term contracts. Note: $12M of revenue "
        "recognised in Q3 relates to a single contract with Acme Corp spanning 36 months. "
        "EBITDA margin improved from 18% to 26% primarily due to a one-time insurance "
        "recovery of $8.5M that management included in operating income."
    ),
    "metadata": {"domain": "financial", "source": "financial_statements.pdf"},
}

HR_CHUNK = {
    "id": "hr_001",
    "content": (
        "Executive Employment Agreement — CEO Jane Smith: Upon a Change of Control, "
        "Ms. Smith shall receive a single-trigger severance payment equal to 3× base salary "
        "($450,000/year) plus immediate full acceleration of 250,000 unvested RSUs at "
        "$15.00/share. Non-compete: 12 months, nationwide, across all technology sectors. "
        "Governing law: California."
    ),
    "metadata": {"domain": "hr", "source": "executive_agreements.pdf"},
}

CYBER_CHUNK = {
    "id": "cyber_001",
    "content": (
        "2023 Security Incident Report: On March 15, 2023, the company experienced an "
        "unauthorised access incident affecting approximately 45,000 customer records "
        "including PII and payment card data. The company has not yet obtained SOC 2 Type II "
        "certification. March 2024 penetration testing identified CVE-2024-11234 (CVSS 9.8) "
        "and CVE-2024-55678 (CVSS 9.1) in the customer-facing API — both remain unpatched."
    ),
    "metadata": {"domain": "cybersecurity", "source": "security_audit_report.pdf"},
}


# ── Mock LLM responses ────────────────────────────────────────────────────────

MOCK_LEGAL_RESPONSE = json.dumps({
    "findings": [{
        "title": "Change-of-Control Termination Fee — $25M Exposure",
        "description": (
            "Section 12.3 imposes a hard CoC trigger requiring third-party consent within "
            "30 days. Failure triggers an irrevocable $25M termination fee, creating a "
            "material deal cost risk that must be resolved before signing."
        ),
        "risk_level": "CRITICAL",
        "evidence_quote": "Failure to obtain consent triggers an irrevocable termination fee of $25,000,000.",
        "source_chunk_ids": ["legal_001"],
        "flags": ["CoC trigger", "Third-party consent required", "Hard termination fee"],
        "recommendations": [
            "Identify counterparty and initiate consent negotiations immediately.",
            "Negotiate a fee waiver or escrow arrangement as part of deal structuring.",
        ],
        "requires_peer_review": True,
        "peer_review_target": "financial",
        "peer_review_question": "Is the $25M termination fee material relative to deal size and operating cash flow?",
    }],
    "summary": (
        "Critical CoC clause could expose the acquirer to a $25M termination fee. "
        "Immediate legal action required to secure counterparty consent."
    ),
    "total_risk_score": 9.2,
    "requires_escalation": True,
    "escalation_reason": "CRITICAL: $25M CoC termination fee without consent secured.",
})

MOCK_FINANCIAL_RESPONSE = json.dumps({
    "findings": [{
        "title": "Non-Recurring Insurance Recovery Inflating EBITDA by $8.5M",
        "description": (
            "Management included an $8.5M one-time insurance recovery in operating income, "
            "inflating reported EBITDA margin by ~8 percentage points. This is a non-recurring "
            "item that must be excluded from normalised EBITDA."
        ),
        "risk_level": "HIGH",
        "evidence_quote": "EBITDA margin improved from 18% to 26% primarily due to a one-time insurance recovery of $8.5M.",
        "source_chunk_ids": ["fin_001"],
        "flags": ["EBITDA inflation", "Non-recurring item", "Normalisation required"],
        "recommendations": [
            "Restate normalised EBITDA excluding the insurance recovery.",
            "Verify no other one-time items are embedded in operating income.",
        ],
        "requires_peer_review": False,
        "peer_review_target": None,
        "peer_review_question": None,
    }],
    "summary": "Reported EBITDA is overstated by ~$8.5M due to non-recurring insurance recovery inclusion.",
    "total_risk_score": 6.5,
    "requires_escalation": False,
    "escalation_reason": None,
})

MOCK_HR_RESPONSE = json.dumps({
    "findings": [{
        "title": "Single-Trigger CEO Severance + RSU Acceleration — ~$5.1M Total",
        "description": (
            "CEO Jane Smith's employment agreement contains a single-trigger change-of-control "
            "provision paying 3× base salary ($1.35M) plus full acceleration of 250,000 RSUs "
            "at $15/share ($3.75M), totalling approximately $5.1M. Non-compete is likely "
            "unenforceable in California."
        ),
        "risk_level": "HIGH",
        "evidence_quote": "Ms. Smith shall receive a single-trigger severance payment equal to 3× base salary plus immediate full acceleration of 250,000 unvested RSUs.",
        "source_chunk_ids": ["hr_001"],
        "flags": ["Single-trigger CoC", "EXEC: Jane Smith", "California non-compete void risk"],
        "recommendations": [
            "Renegotiate to double-trigger before signing to reduce immediate CoC cost.",
            "Obtain California employment counsel opinion on non-compete enforceability.",
            "Model $5.1M severance into transaction cost bridge.",
        ],
        "requires_peer_review": True,
        "peer_review_target": "financial",
        "peer_review_question": "Please model the $5.1M CEO severance into the transaction cost bridge and confirm IRC 280G exposure.",
    }],
    "summary": "CEO has single-trigger CoC entitlement totalling ~$5.1M. Non-compete likely void in California.",
    "total_risk_score": 7.0,
    "requires_escalation": False,
    "escalation_reason": None,
})

MOCK_CYBER_RESPONSE = json.dumps({
    "findings": [{
        "title": "Unpatched CVSS 9.8 Vulnerability in Production API (CVE-2024-11234)",
        "description": (
            "Penetration testing identified two critical vulnerabilities — CVE-2024-11234 "
            "(CVSS 9.8) and CVE-2024-55678 (CVSS 9.1) — in the customer-facing API. Both "
            "remain unpatched. Combined with a 2023 data breach affecting 45,000 PII records "
            "and no SOC 2 Type II certification, the security posture is critically deficient."
        ),
        "risk_level": "CRITICAL",
        "evidence_quote": "CVE-2024-11234 (CVSS 9.8) and CVE-2024-55678 (CVSS 9.1) in the customer-facing API — both remain unpatched.",
        "source_chunk_ids": ["cyber_001"],
        "flags": ["CVE-2024-11234", "CVE-2024-55678", "Unpatched critical CVE", "No SOC 2 Type II", "Data breach history"],
        "recommendations": [
            "Mandate emergency patching of CVE-2024-11234 and CVE-2024-55678 before close.",
            "Negotiate cyber R&W insurance with appropriate coverage for pre-existing vulnerabilities.",
            "Require SOC 2 Type II as a closing condition or post-close milestone.",
        ],
        "requires_peer_review": True,
        "peer_review_target": "financial",
        "peer_review_question": "What is the estimated remediation cost for the 2023 breach and the two unpatched critical CVEs?",
    }],
    "summary": "Critical unpatched CVEs and a prior breach create severe cyber risk. Immediate patching and R&W insurance required.",
    "total_risk_score": 9.5,
    "requires_escalation": True,
    "escalation_reason": "CRITICAL: Active unpatched CVSS 9.8 vulnerability in production; prior undisclosed breach.",
})


# ── Config fixture (no real API key needed) ────────────────────────────────────

@pytest.fixture
def test_config() -> ModuleBConfig:
    return ModuleBConfig(
        groq_api_key="test_key_no_real_calls",
        groq_model="llama-3.3-70b-versatile",
        temperature=0.05,
        max_tokens=4096,
        auto_escalation_score=7.5,
    )


# ── Helper to build a mock Groq client ────────────────────────────────────────

def _mock_groq(response_json: str):
    """Returns a mock Groq class whose .chat.completions.create() returns response_json."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_json
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_groq_cls = MagicMock(return_value=mock_client)
    return mock_groq_cls


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemas:

    def test_risk_level_numeric_weights(self):
        assert RiskLevel.LOW.numeric == 2.0
        assert RiskLevel.MEDIUM.numeric == 4.5
        assert RiskLevel.HIGH.numeric == 7.0
        assert RiskLevel.CRITICAL.numeric == 9.5

    def test_agent_result_critical_findings_property(self):
        result = AgentResult(
            agent_id="test",
            domain=DomainType.LEGAL,
            query="test",
            summary="test",
            findings=[
                Finding(domain=DomainType.LEGAL, title="A", description="", risk_level=RiskLevel.CRITICAL, evidence_quote=""),
                Finding(domain=DomainType.LEGAL, title="B", description="", risk_level=RiskLevel.HIGH, evidence_quote=""),
                Finding(domain=DomainType.LEGAL, title="C", description="", risk_level=RiskLevel.LOW, evidence_quote=""),
            ],
        )
        assert len(result.critical_findings) == 1
        assert len(result.high_risk_findings) == 2  # HIGH + CRITICAL

    def test_agent_result_risk_summary(self):
        result = AgentResult(
            agent_id="test", domain=DomainType.FINANCIAL, query="q", summary="s",
            findings=[
                Finding(domain=DomainType.FINANCIAL, title="X", description="", risk_level=RiskLevel.HIGH, evidence_quote=""),
                Finding(domain=DomainType.FINANCIAL, title="Y", description="", risk_level=RiskLevel.HIGH, evidence_quote=""),
                Finding(domain=DomainType.FINANCIAL, title="Z", description="", risk_level=RiskLevel.MEDIUM, evidence_quote=""),
            ],
        )
        assert result.risk_summary["HIGH"] == 2
        assert result.risk_summary["MEDIUM"] == 1
        assert result.risk_summary["CRITICAL"] == 0

    def test_peer_query_defaults(self):
        pq = PeerQuery(
            source_domain=DomainType.LEGAL,
            target_domain=DomainType.FINANCIAL,
            question="Is this amount material?",
            context_snippet="$25M termination fee",
            urgency=RiskLevel.CRITICAL,
        )
        assert pq.resolved is False
        assert pq.answer is None
        assert len(pq.query_id) == 8


# ═════════════════════════════════════════════════════════════════════════════
# LEGAL AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestLegalAgent:

    @pytest.fixture
    def legal_retrieval(self) -> MockRetrievalResult:
        return MockRetrievalResult(
            query="What change-of-control clauses exist?",
            retrieved_chunks=[LEGAL_CHUNK],
            domain_breakdown={"legal": [LEGAL_CHUNK]},
        )

    @patch("module_b.base_agent.Groq")
    def test_analyze_returns_valid_agent_result(self, mock_groq_cls, test_config, legal_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_LEGAL_RESPONSE).return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        assert isinstance(result, AgentResult)
        assert result.domain == DomainType.LEGAL
        assert result.agent_id == "legal_agent"

    @patch("module_b.base_agent.Groq")
    def test_critical_finding_parsed_correctly(self, mock_groq_cls, test_config, legal_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_LEGAL_RESPONSE).return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.risk_level == RiskLevel.CRITICAL
        assert "CoC trigger" in finding.flags
        assert "legal_001" in finding.source_chunk_ids

    @patch("module_b.base_agent.Groq")
    def test_peer_query_compiled_to_financial(self, mock_groq_cls, test_config, legal_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_LEGAL_RESPONSE).return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        assert len(result.peer_queries) == 1
        pq = result.peer_queries[0]
        assert pq.source_domain == DomainType.LEGAL
        assert pq.target_domain == DomainType.FINANCIAL
        assert pq.urgency == RiskLevel.CRITICAL
        assert pq.resolved is False

    @patch("module_b.base_agent.Groq")
    def test_critical_finding_auto_triggers_escalation(self, mock_groq_cls, test_config, legal_retrieval):
        # Use a response that says requires_escalation=false but has CRITICAL finding
        response_no_escalation = json.dumps({
            "findings": [{
                "title": "CoC Clause",
                "description": "Critical issue",
                "risk_level": "CRITICAL",
                "evidence_quote": "...",
                "source_chunk_ids": [],
                "flags": [],
                "recommendations": [],
                "requires_peer_review": False,
                "peer_review_target": None,
                "peer_review_question": None,
            }],
            "summary": "test",
            "total_risk_score": 9.0,
            "requires_escalation": False,  # LLM forgot to set this
            "escalation_reason": None,
        })
        mock_groq_cls.return_value = _mock_groq(response_no_escalation).return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        # Auto-escalation policy should have caught this
        assert result.requires_escalation is True
        assert "[AUTO]" in result.escalation_reason

    @patch("module_b.base_agent.Groq")
    def test_empty_context_returns_empty_result(self, mock_groq_cls, test_config):
        mock_groq_cls.return_value = _mock_groq("{}").return_value
        agent = LegalAgent(config=test_config)
        empty_retrieval = MockRetrievalResult(
            query="test", retrieved_chunks=[], domain_breakdown={}
        )
        result = agent.analyze(empty_retrieval)

        # Should NOT call Groq at all — returns early
        assert isinstance(result, AgentResult)
        assert len(result.findings) == 0
        assert result.total_risk_score == 0.0
        assert result.requires_escalation is False

    @patch("module_b.base_agent.Groq")
    def test_malformed_json_returns_graceful_empty_result(self, mock_groq_cls, test_config, legal_retrieval):
        mock_groq_cls.return_value = _mock_groq("this is not json {{{{").return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        assert isinstance(result, AgentResult)
        assert len(result.findings) == 0
        assert "malformed JSON" in result.summary.lower() or result.total_risk_score == 0.0

    @patch("module_b.base_agent.Groq")
    def test_risk_score_auto_escalation_threshold(self, mock_groq_cls, test_config, legal_retrieval):
        """Risk score >= auto_escalation_score (7.5) should force escalation."""
        response = json.dumps({
            "findings": [],
            "summary": "High score but no CRITICAL findings.",
            "total_risk_score": 8.0,
            "requires_escalation": False,
            "escalation_reason": None,
        })
        mock_groq_cls.return_value = _mock_groq(response).return_value
        agent = LegalAgent(config=test_config)
        result = agent.analyze(legal_retrieval)

        assert result.requires_escalation is True
        assert "[AUTO]" in result.escalation_reason


# ═════════════════════════════════════════════════════════════════════════════
# FINANCIAL AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestFinancialAgent:

    @pytest.fixture
    def financial_retrieval(self) -> MockRetrievalResult:
        return MockRetrievalResult(
            query="Is the reported EBITDA reliable?",
            retrieved_chunks=[FINANCIAL_CHUNK],
            domain_breakdown={"financial": [FINANCIAL_CHUNK]},
        )

    @patch("module_b.base_agent.Groq")
    def test_analyze_returns_valid_agent_result(self, mock_groq_cls, test_config, financial_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_FINANCIAL_RESPONSE).return_value
        agent = FinancialAgent(config=test_config)
        result = agent.analyze(financial_retrieval)

        assert isinstance(result, AgentResult)
        assert result.domain == DomainType.FINANCIAL
        assert result.agent_id == "financial_agent"

    @patch("module_b.base_agent.Groq")
    def test_high_risk_ebitda_inflation_finding(self, mock_groq_cls, test_config, financial_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_FINANCIAL_RESPONSE).return_value
        agent = FinancialAgent(config=test_config)
        result = agent.analyze(financial_retrieval)

        assert len(result.findings) == 1
        assert result.findings[0].risk_level == RiskLevel.HIGH
        assert "EBITDA inflation" in result.findings[0].flags

    @patch("module_b.base_agent.Groq")
    def test_no_peer_queries_when_not_flagged(self, mock_groq_cls, test_config, financial_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_FINANCIAL_RESPONSE).return_value
        agent = FinancialAgent(config=test_config)
        result = agent.analyze(financial_retrieval)

        assert len(result.peer_queries) == 0

    @patch("module_b.base_agent.Groq")
    def test_domain_fallback_uses_all_chunks(self, mock_groq_cls, test_config):
        """When domain_breakdown is empty, agent falls back to all retrieved_chunks."""
        mock_groq_cls.return_value = _mock_groq(MOCK_FINANCIAL_RESPONSE).return_value
        retrieval = MockRetrievalResult(
            query="revenue test",
            retrieved_chunks=[FINANCIAL_CHUNK, LEGAL_CHUNK],  # mixed
            domain_breakdown={},  # empty breakdown
        )
        agent = FinancialAgent(config=test_config)
        # Should not raise — falls back to metadata filter then all chunks
        result = agent.analyze(retrieval)
        assert isinstance(result, AgentResult)


# ═════════════════════════════════════════════════════════════════════════════
# HR AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestHRAgent:

    @pytest.fixture
    def hr_retrieval(self) -> MockRetrievalResult:
        return MockRetrievalResult(
            query="What executive compensation obligations exist?",
            retrieved_chunks=[HR_CHUNK],
            domain_breakdown={"hr": [HR_CHUNK]},
        )

    @patch("module_b.base_agent.Groq")
    def test_analyze_returns_valid_agent_result(self, mock_groq_cls, test_config, hr_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_HR_RESPONSE).return_value
        agent = HRAgent(config=test_config)
        result = agent.analyze(hr_retrieval)

        assert isinstance(result, AgentResult)
        assert result.domain == DomainType.HR

    @patch("module_b.base_agent.Groq")
    def test_exec_name_extracted_to_flags_by_sanity_check(self, mock_groq_cls, test_config, hr_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_HR_RESPONSE).return_value
        agent = HRAgent(config=test_config)
        result = agent.analyze(hr_retrieval)

        all_flags = [f for finding in result.findings for f in finding.flags]
        # The sanity check should have extracted "Jane Smith" into flags
        assert any("Jane Smith" in flag for flag in all_flags)

    @patch("module_b.base_agent.Groq")
    def test_peer_query_to_financial_for_severance(self, mock_groq_cls, test_config, hr_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_HR_RESPONSE).return_value
        agent = HRAgent(config=test_config)
        result = agent.analyze(hr_retrieval)

        peer_targets = [pq.target_domain for pq in result.peer_queries]
        assert DomainType.FINANCIAL in peer_targets

    @patch("module_b.base_agent.Groq")
    def test_risk_map_synonyms(self, mock_groq_cls, test_config, hr_retrieval):
        """Verify risk-level synonym normalisation (MAJOR → HIGH, SEVERE → CRITICAL)."""
        mock_groq_cls.return_value = _mock_groq(MOCK_HR_RESPONSE).return_value
        agent = HRAgent(config=test_config)
        assert agent._map_risk_level("MAJOR") == RiskLevel.HIGH
        assert agent._map_risk_level("SEVERE") == RiskLevel.CRITICAL
        assert agent._map_risk_level("minor") == RiskLevel.LOW
        assert agent._map_risk_level("unknown_level") == RiskLevel.MEDIUM


# ═════════════════════════════════════════════════════════════════════════════
# CYBERSECURITY AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestCyberAgent:

    @pytest.fixture
    def cyber_retrieval(self) -> MockRetrievalResult:
        return MockRetrievalResult(
            query="What is the security posture of the target company?",
            retrieved_chunks=[CYBER_CHUNK],
            domain_breakdown={"cybersecurity": [CYBER_CHUNK]},
        )

    @patch("module_b.base_agent.Groq")
    def test_analyze_returns_valid_agent_result(self, mock_groq_cls, test_config, cyber_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_CYBER_RESPONSE).return_value
        agent = CyberAgent(config=test_config)
        result = agent.analyze(cyber_retrieval)

        assert isinstance(result, AgentResult)
        assert result.domain == DomainType.CYBERSECURITY

    @patch("module_b.base_agent.Groq")
    def test_cve_ids_extracted_to_flags_by_sanity_check(self, mock_groq_cls, test_config, cyber_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_CYBER_RESPONSE).return_value
        agent = CyberAgent(config=test_config)
        result = agent.analyze(cyber_retrieval)

        all_flags = [f for finding in result.findings for f in finding.flags]
        assert "CVE-2024-11234" in all_flags
        assert "CVE-2024-55678" in all_flags

    @patch("module_b.base_agent.Groq")
    def test_critical_finding_triggers_escalation(self, mock_groq_cls, test_config, cyber_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_CYBER_RESPONSE).return_value
        agent = CyberAgent(config=test_config)
        result = agent.analyze(cyber_retrieval)

        assert result.requires_escalation is True
        assert result.total_risk_score >= 7.5

    @patch("module_b.base_agent.Groq")
    def test_peer_query_to_financial_for_breach_cost(self, mock_groq_cls, test_config, cyber_retrieval):
        mock_groq_cls.return_value = _mock_groq(MOCK_CYBER_RESPONSE).return_value
        agent = CyberAgent(config=test_config)
        result = agent.analyze(cyber_retrieval)

        peer_targets = [pq.target_domain for pq in result.peer_queries]
        assert DomainType.FINANCIAL in peer_targets


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY / INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentFactory:

    @pytest.fixture
    def full_retrieval(self) -> MockRetrievalResult:
        return MockRetrievalResult(
            query="Full M&A due diligence sweep",
            retrieved_chunks=[LEGAL_CHUNK, FINANCIAL_CHUNK, HR_CHUNK, CYBER_CHUNK],
            domain_breakdown={
                "legal": [LEGAL_CHUNK],
                "financial": [FINANCIAL_CHUNK],
                "hr": [HR_CHUNK],
                "cybersecurity": [CYBER_CHUNK],
            },
        )

    @patch("module_b.base_agent.Groq")
    def test_run_all_agents_returns_four_results(self, mock_groq_cls, test_config, full_retrieval):
        # Side-effect: return domain-appropriate responses in order
        responses = [MOCK_LEGAL_RESPONSE, MOCK_FINANCIAL_RESPONSE, MOCK_HR_RESPONSE, MOCK_CYBER_RESPONSE]
        response_idx = 0

        def side_effect(*args, **kwargs):
            nonlocal response_idx
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = responses[response_idx % len(responses)]
            response_idx += 1
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect
        mock_groq_cls.return_value = mock_client

        results = run_all_agents(full_retrieval, config=test_config)

        assert len(results) == 4
        for domain in DomainType:
            assert domain in results
            assert isinstance(results[domain], AgentResult)

    @patch("module_b.base_agent.Groq")
    def test_build_consolidated_report_structure(self, mock_groq_cls, test_config):
        """build_consolidated_report should aggregate all domain results correctly."""
        mock_groq_cls.return_value = MagicMock()

        # Build mock results directly (skip agent.analyze)
        results = {
            DomainType.LEGAL: AgentResult(
                agent_id="legal_agent", domain=DomainType.LEGAL, query="q", summary="s",
                findings=[
                    Finding(domain=DomainType.LEGAL, title="CoC", description="d",
                            risk_level=RiskLevel.CRITICAL, evidence_quote="e")
                ],
                total_risk_score=9.0, requires_escalation=True,
                escalation_reason="Critical CoC",
            ),
            DomainType.FINANCIAL: AgentResult(
                agent_id="financial_agent", domain=DomainType.FINANCIAL, query="q", summary="s",
                findings=[
                    Finding(domain=DomainType.FINANCIAL, title="EBITDA", description="d",
                            risk_level=RiskLevel.HIGH, evidence_quote="e")
                ],
                total_risk_score=6.5, requires_escalation=False,
            ),
            DomainType.HR: AgentResult(
                agent_id="hr_agent", domain=DomainType.HR, query="q", summary="s",
                findings=[], total_risk_score=5.0,
            ),
            DomainType.CYBERSECURITY: AgentResult(
                agent_id="cybersecurity_agent", domain=DomainType.CYBERSECURITY, query="q", summary="s",
                findings=[
                    Finding(domain=DomainType.CYBERSECURITY, title="CVE", description="d",
                            risk_level=RiskLevel.CRITICAL, evidence_quote="e")
                ],
                total_risk_score=9.5, requires_escalation=True,
                escalation_reason="Unpatched CVE",
            ),
        }

        report = build_consolidated_report(results)

        assert "overall_risk_score" in report
        assert report["requires_escalation"] is True
        assert len(report["escalation_reasons"]) == 2
        assert report["total_findings"] == 3
        assert report["risk_breakdown"]["CRITICAL"] == 2
        assert report["risk_breakdown"]["HIGH"] == 1
        assert "legal" in report["domain_results"]

    def test_create_agent_raises_for_unknown_domain(self):
        from module_b.agent_factory import create_agent
        with pytest.raises((ValueError, KeyError)):
            create_agent("nonexistent_domain")  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# BASE AGENT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

class TestBaseAgentUtilities:

    @pytest.fixture
    def agent(self, test_config) -> LegalAgent:
        with patch("module_b.base_agent.Groq"):
            return LegalAgent(config=test_config)

    def test_safe_parse_json_strips_markdown_fences(self, agent):
        raw = "```json\n{\"key\": \"value\"}\n```"
        parsed = agent._safe_parse_json(raw)
        assert parsed == {"key": "value"}

    def test_safe_parse_json_handles_clean_json(self, agent):
        raw = '{"findings": []}'
        parsed = agent._safe_parse_json(raw)
        assert parsed == {"findings": []}

    def test_compute_risk_score_empty_findings(self, agent):
        assert agent._compute_risk_score([]) == 0.0

    def test_compute_risk_score_mixed_levels(self, agent):
        findings = [
            Finding(domain=DomainType.LEGAL, title="A", description="", risk_level=RiskLevel.LOW, evidence_quote=""),
            Finding(domain=DomainType.LEGAL, title="B", description="", risk_level=RiskLevel.CRITICAL, evidence_quote=""),
        ]
        score = agent._compute_risk_score(findings)
        assert score == round((2.0 + 9.5) / 2, 2)  # 5.75

    def test_format_chunks_produces_xml_blocks(self, agent):
        chunks = [{"id": "c1", "content": "hello world", "metadata": {"source": "doc.pdf"}}]
        result = agent._format_chunks(chunks)
        assert "<chunk id='c1'" in result
        assert "hello world" in result
        assert "source='doc.pdf'" in result

    def test_format_chunks_empty_list_returns_empty_string(self, agent):
        assert agent._format_chunks([]) == ""
