"""Shared fixtures. All documents under tests/fixtures are FAKE and exist only for tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # for pdf_helper

from complibot.answer import CompliBot  # noqa: E402
from complibot.config import Settings  # noqa: E402
from complibot.embeddings import HashEmbedder  # noqa: E402
from complibot.ingest import build_index  # noqa: E402
from complibot.llm import FakeProvider  # noqa: E402
from complibot.retrieve import HybridRetriever  # noqa: E402

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=FIXTURE_DATA, index_dir=tmp_path / "index", top_k=4)


@pytest.fixture
def index(settings):
    return build_index(settings)


@pytest.fixture
def retriever(index, settings) -> HybridRetriever:
    return HybridRetriever(index, settings)


@pytest.fixture
def hybrid_retriever(settings) -> HybridRetriever:
    s = settings.with_(embedder="hash")
    emb = HashEmbedder()
    return HybridRetriever(build_index(s, embedder=emb), s, embedder=emb)


@pytest.fixture
def bot(retriever, settings) -> CompliBot:
    return CompliBot(settings, retriever, FakeProvider())
