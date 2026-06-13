"""
module_a/tests/conftest.py
───────────────────────────
Shared pytest fixtures for Module A tests.

The session-scoped `pipeline_with_data` fixture:
  1. Writes synthetic M&A documents to a temp directory.
  2. Creates a fresh RAGPipeline (isolated ChromaDB + BM25 store in tmp dirs).
  3. Ingests all synthetic docs.
  4. Yields the pipeline for the full test session, then cleans up.

This matches the spec's requirement to test against realistic M&A content
without needing a live data room.
"""

import os
import tempfile
import pytest

# ── Synthetic M&A Data Room ───────────────────────────────────────────────────
# Each document is crafted to test a specific domain and retrieval scenario.

SYNTHETIC_DOCS: dict[str, str] = {
    # ── Legal ──────────────────────────────────────────────────────────────
    "acquisition_agreement.txt": """
    MERGER AGREEMENT — ARTICLE 12: CHANGE OF CONTROL PROVISIONS
    Section 12.1: In the event of a Change of Control, all outstanding unvested
    stock options and RSUs shall immediately vest and become exercisable.
    Section 12.2: Any acquiring entity must assume all existing employment
    agreements within 30 calendar days of closing. Failure to do so triggers
    mandatory arbitration under Section 15.4 of this Agreement.
    Section 12.3: The Target Company shall pay a termination fee of $18.5M
    if this Agreement is terminated due to a competing acquisition offer.
    Section 12.4: Change of Control is defined as any transaction resulting in
    a transfer of more than 50% of outstanding voting shares to a third party.
    """,

    "customer_contracts.txt": """
    ENTERPRISE CUSTOMER CONTRACT — SCHEDULE A: ASSIGNMENT CLAUSE
    Article 7 — Assignment: This Agreement may not be assigned by either party
    without the prior written consent of the other party, which shall not be
    unreasonably withheld. Notwithstanding the foregoing, either party may
    assign this Agreement to a successor entity in connection with a merger,
    acquisition, or sale of all or substantially all of its assets.
    Penalty for breach of assignment clause: liquidated damages of $500,000
    per incident plus reasonable attorneys' fees.
    """,

    # ── Financial ─────────────────────────────────────────────────────────
    "financial_audit_2023.txt": """
    CONSOLIDATED REVENUE STATEMENT — FISCAL YEAR 2023
    Total Revenue: $124.7M (18% year-over-year growth)
    Recurring Revenue (ARR): $98.2M | One-time: $26.5M
    Revenue Concentration: Top 5 clients = 67% of total revenue.
    Client #1 (Acme Corp): $31.2M (25% of revenue) — renewal risk flagged.
    EBITDA: $29.2M | EBITDA Margin: 23.4%
    Notable: Q3 revenue dip of $4.1M due to one-time client churn.
    Gross margin declined 2.1% YoY due to increased cloud infrastructure costs.
    Net Debt: $12.4M. Working capital: $18.7M. CapEx: $3.2M.
    """,

    "tax_exposure_report.txt": """
    TAX DUE DILIGENCE SUMMARY — FY2021–FY2023
    Transfer pricing arrangements with two offshore subsidiaries (Ireland, Singapore)
    have not been formally documented. Estimated potential liability: $4.2M–$7.8M.
    R&D tax credits claimed: $6.1M over 3 years. Documentation adequate.
    Sales tax nexus established in 14 states; 3 states (TX, NY, WA) have
    outstanding assessments totalling $890K currently under appeal.
    Deferred revenue balance: $22.3M — timing difference between GAAP and cash.
    """,

    # ── HR ─────────────────────────────────────────────────────────────────
    "hr_headcount_report.txt": """
    WORKFORCE SUMMARY — Q4 2023
    Total Headcount: 847 full-time employees across 12 countries.
    Department Breakdown: Engineering 312 | Sales 198 | Operations 245 | G&A 92
    Average employee tenure: 3.2 years.
    Voluntary attrition rate: 14.2% (industry median: 11.4%) — elevated risk.
    Key Person Risk: 3 C-suite executives hold critical customer relationships
    with no documented succession plans. Departure triggers retention clauses
    totalling $4.7M in guaranteed bonuses payable within 90 days.
    """,

    "compensation_summary.txt": """
    EXECUTIVE COMPENSATION & EQUITY SCHEDULE
    CEO total compensation: $2.4M (base $650K, target bonus $450K, equity $1.3M)
    CTO total compensation: $1.9M (base $550K, target bonus $375K, equity $975K)
    Outstanding unvested options: 4.2M shares at avg strike $8.40
    Change-in-control acceleration: Double-trigger for all equity grants.
    Severance obligations on involuntary termination post-acquisition: $3.1M.
    Non-compete agreements: 18-month restriction for all VP+ employees.
    """,

    # ── Cybersecurity ──────────────────────────────────────────────────────
    "cybersecurity_assessment.txt": """
    INFORMATION SECURITY ASSESSMENT — EXTERNAL PENTEST RESULTS (Oct 2023)
    Overall Risk Rating: HIGH
    Critical Findings:
      CVE-2023-4521: Unpatched SQL injection vulnerability in customer-facing API.
        Risk: Unauthorized access to customer PII data. CVSS Score: 9.1.
      CVE-2023-7813: Server-side request forgery (SSRF) in internal admin portal.
        Risk: Internal network reconnaissance. CVSS Score: 8.7.
    Compliance Status: SOC 2 Type II certification expired January 2024.
    Cloud Infrastructure: AWS us-east-1. No data residency controls for EU customers.
    Estimated remediation cost: $340K–$480K. Timeline: 4–6 months.
    """,

    "data_governance_policy.txt": """
    DATA GOVERNANCE & PRIVACY COMPLIANCE POLICY
    GDPR compliance status: Partial. Data Processing Agreements (DPAs) missing
    for 7 of 23 EU-based sub-processors. Potential regulatory exposure: up to
    4% of global annual turnover (~$4.99M based on FY2023 revenue).
    CCPA: Compliant. Privacy notice updated January 2024.
    Data retention policy: 7 years for financial records, 3 years for operational.
    Backup encryption: AES-256 at rest. In-transit: TLS 1.3.
    Incident response plan: Last tested 18 months ago. Requires update.
    """,
}


@pytest.fixture(scope="session")
def pipeline_with_data():
    """
    Session-scoped fixture: creates an isolated RAGPipeline with synthetic
    M&A documents ingested.  Cleans up temp directories on teardown.
    """
    import sys
    # Allow importing module_a from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    with tempfile.TemporaryDirectory() as data_room_dir, \
         tempfile.TemporaryDirectory() as chroma_dir, \
         tempfile.TemporaryDirectory() as bm25_dir:

        # Write synthetic docs
        for filename, content in SYNTHETIC_DOCS.items():
            with open(os.path.join(data_room_dir, filename), "w") as f:
                f.write(content)

        # Point config at isolated temp storage so tests don't pollute real data
        from module_a.config import cfg
        cfg.chroma_persist_dir = chroma_dir
        cfg.bm25_index_path    = os.path.join(bm25_dir, "bm25_test.pkl")

        from module_a.pipeline import RAGPipeline
        # Reset singletons so each test session gets fresh model instances
        from module_a.retrieval.embedder import BGE_M3_Embedder
        from module_a.reranker import CrossEncoderReranker
        BGE_M3_Embedder._instance     = None
        CrossEncoderReranker._instance = None

        pipeline = RAGPipeline()
        pipeline.ingest(data_room_dir)
        yield pipeline