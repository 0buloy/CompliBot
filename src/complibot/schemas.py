"""Core data types shared across ingestion, retrieval, answering and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    """One retrievable unit of text. For section chunking this is one clause."""

    chunk_id: str  # unique, stable: "<doc_id>::<section_id>[#part]"
    doc_id: str  # file stem, e.g. "mas-trm-guidelines-2021"
    doc_title: str  # human title, e.g. "MAS Technology Risk Management Guidelines (2021)"
    section_id: str  # clause number, e.g. "4.2.1" ("preamble" before the first heading)
    section_title: str  # nearest heading title, e.g. "Technology Risk Management Strategy"
    page_start: int  # 1-indexed
    page_end: int
    access_level: str  # "public" | "internal" (from the folder the file lives in)
    text: str
    # Clause IDs whose text overlaps this chunk. For section chunks this is
    # [section_id]; for fixed-size chunks it can be several. Used by recall@k.
    covers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)

    @property
    def citation_label(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"{self.doc_title}, s.{self.section_id} ({pages})"


@dataclass
class Retrieved:
    chunk: Chunk
    score: float  # fused (RRF) score, or reranker score
    bm25_rank: int | None = None  # 1-indexed rank in the BM25 list, None if absent
    dense_rank: int | None = None


# ---------- API / answer models (pydantic so FastAPI can serialise them) ----------


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    snippet: str = Field(description="First ~200 characters of the cited clause")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""  # the model that actually served the request (may differ on fallback)
    llm_latency_ms: float = 0.0


AnswerStatus = Literal["answered", "refused", "unsupported", "blocked", "error"]


class AnswerResult(BaseModel):
    question: str
    answer: str
    status: AnswerStatus
    refused: bool
    supported: bool = Field(description="True only if every citation maps to a retrieved chunk")
    citations: list[Citation] = []
    invalid_citations: list[str] = Field(default=[], description="Citations the LLM made up")
    retrieved_ids: list[str] = []
    role: str = "public"
    latency_ms: float = 0.0
    usage: Usage = Usage()
    error: str | None = None
    pii_redacted: dict[str, int] = Field(default={}, description="Counts of PII entities redacted from the input")
    related_clauses: list[Citation] = Field(default=[], description="Top clauses, returned when the LLM is down")
