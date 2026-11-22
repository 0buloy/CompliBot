"""Runtime configuration, read from environment variables (optionally a .env file).

Everything tunable lives here so experiments (Phase 3) can change behaviour
without touching code, and so no secret is ever hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Which document access levels each role may read. The folder a file sits in
# (data/raw/public or data/raw/internal) decides its access level.
ROLE_ACCESS: dict[str, frozenset[str]] = {
    "public": frozenset({"public"}),
    "internal": frozenset({"public", "internal"}),
}
ACCESS_LEVELS = ("public", "internal")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")

    # Chunking
    chunking: str = "section"  # "section" | "fixed"
    max_chunk_chars: int = 2400  # long clauses get split into parts
    fixed_chunk_chars: int = 1200
    fixed_overlap_chars: int = 200

    # Retrieval
    top_k: int = 6
    embedder: str = "none"  # "none" | "openai" | "hash"
    rrf_k: int = 60
    candidate_pool: int = 30  # how many results each retriever contributes to RRF
    rewriter: str = "none"  # "none" | "rules" | "llm"
    reranker: str = "none"  # "none" | "lexical" | "cross-encoder" | "llm"
    rerank_pool: int = 20  # first-stage candidates the reranker re-orders
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    llm_provider: str = "fake"  # "fake" | "anthropic" | "openai"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"
    anthropic_fallbacks: str = "default"
    openai_model: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    max_output_tokens: int = 4000

    # Evaluation: the judge should be a different model from the one under test
    judge_provider: str = "fake"  # "fake" | "anthropic" | "openai"
    judge_model: str = "claude-sonnet-5-5"

    # Guardrails & reliability
    pii_engine: str = "auto"  # "auto" (Presidio if available, else regex) | "presidio" | "regex" | "off"
    pii_person: bool = False  # also redact PERSON names (spaCy NER; more false positives)
    spacy_model: str = "en_core_web_sm"
    block_injection: bool = True
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3  # SDK-level retries with exponential backoff (429/5xx/connection)
    agent_max_steps: int = 6
    audit_log: str = ""  # path to a JSONL audit log; empty = off

    # API access control
    internal_api_keys: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls) -> "Settings":
        keys = frozenset(k.strip() for k in _env("COMPLIBOT_INTERNAL_API_KEYS").split(",") if k.strip())
        return cls(
            data_dir=Path(_env("COMPLIBOT_DATA_DIR", "data/raw")),
            index_dir=Path(_env("COMPLIBOT_INDEX_DIR", "data/index")),
            chunking=_env("COMPLIBOT_CHUNKING", "section"),
            top_k=_env_int("COMPLIBOT_TOP_K", 6),
            embedder=_env("COMPLIBOT_EMBEDDER", "none"),
            rewriter=_env("COMPLIBOT_REWRITER", "none"),
            reranker=_env("COMPLIBOT_RERANKER", "none"),
            rerank_model=_env("COMPLIBOT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            llm_provider=_env("COMPLIBOT_LLM_PROVIDER", "fake"),
            anthropic_model=_env("ANTHROPIC_MODEL", "claude-opus-5-5"),
            anthropic_effort=_env("ANTHROPIC_EFFORT", "medium"),
            anthropic_fallbacks=_env("ANTHROPIC_FALLBACKS", "default"),
            openai_model=_env("OPENAI_MODEL", ""),
            openai_embed_model=_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            max_output_tokens=_env_int("COMPLIBOT_MAX_OUTPUT_TOKENS", 4000),
            judge_provider=_env("COMPLIBOT_JUDGE_PROVIDER", "fake"),
            judge_model=_env("COMPLIBOT_JUDGE_MODEL", "claude-sonnet-5-5"),
            pii_engine=_env("COMPLIBOT_PII_ENGINE", "auto"),
            pii_person=_env("COMPLIBOT_PII_PERSON", "false").lower() in ("1", "true", "yes"),
            spacy_model=_env("COMPLIBOT_SPACY_MODEL", "en_core_web_sm"),
            block_injection=_env("COMPLIBOT_BLOCK_INJECTION", "true").lower() in ("1", "true", "yes"),
            llm_timeout_s=float(_env("COMPLIBOT_LLM_TIMEOUT", "60")),
            llm_max_retries=_env_int("COMPLIBOT_LLM_MAX_RETRIES", 3),
            agent_max_steps=_env_int("COMPLIBOT_AGENT_MAX_STEPS", 6),
            audit_log=_env("COMPLIBOT_AUDIT_LOG", ""),
            internal_api_keys=keys,
        )

    def with_(self, **changes) -> "Settings":
        """Return a copy with some fields changed (handy in tests and experiments)."""
        return replace(self, **changes)


def allowed_levels(role: str) -> frozenset[str]:
    """Access levels a role may read. Unknown roles get the least privilege."""
    return ROLE_ACCESS.get(role, ROLE_ACCESS["public"])
