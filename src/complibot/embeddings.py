"""Pluggable text embedders for the (optional) dense half of hybrid search.

- "none":   dense search disabled, BM25 only (default, fully offline)
- "hash":   feature-hashed bag of words. Not semantic; it exists so tests and CI
            exercise the whole dense code path with no model download or API key.
- "openai": OpenAI embeddings API (model from OPENAI_EMBED_MODEL).

Anthropic has no embeddings endpoint, which is why the dense side is a separate
provider from the LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalised
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _normalise(out)


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 128):
        from openai import OpenAI  # imported lazily: optional dependency

        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            vectors.extend(d.embedding for d in resp.data)
        return _normalise(np.array(vectors, dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder | None:
    kind = settings.embedder.lower()
    if kind in ("", "none", "off"):
        return None
    if kind == "hash":
        return HashEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(settings.openai_embed_model)
    raise ValueError(f"Unknown embedder {settings.embedder!r} (use none | hash | openai)")
