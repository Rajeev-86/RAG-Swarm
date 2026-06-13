"""
module_b/config.py

Module B configuration. Reads from the same .env file as Module A.
Groq is the default backend; swap GROQ_MODEL_B for a larger model on prod.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModuleBConfig:
    # ── Groq ──────────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    # Fast inference: llama-3.1-8b-instant  |  Higher accuracy: llama-3.3-70b-versatile
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL_B", "llama-3.3-70b-versatile")
    )

    # ── Inference hyperparameters ──────────────────────────────────────────
    # Very low temperature — agents do extraction, not generation.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.05"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    )

    # ── Risk thresholds ────────────────────────────────────────────────────
    # AgentResult.total_risk_score >= this → auto-set requires_escalation=True
    auto_escalation_score: float = field(
        default_factory=lambda: float(os.getenv("AUTO_ESCALATION_SCORE", "7.5"))
    )

    @classmethod
    def from_env(cls) -> "ModuleBConfig":
        return cls()

    def validate(self) -> None:
        """Call once at startup to surface missing credentials early."""
        if not self.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "  1. Get a free key at https://console.groq.com/keys\n"
                "  2. Add GROQ_API_KEY=<key> to your .env file"
            )