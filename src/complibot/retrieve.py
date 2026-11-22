"""Hybrid retrieval: BM25 + (optional) dense cosine search, fused with RRF.

Permission-aware by construction: the role's allowed access levels are applied
*before* ranking, so an internal chunk can never be scored, returned, or leak
into the LLM prompt for a public user.

Reciprocal rank fusion (Cormack et al., 2009):
    score(d) = sum over retrievers r of 1 / (k + rank_r(d)),  k = 60
It only uses ranks, so we never have to calibrate BM25 scores against cosine
similarities, which live on completely different scales.
"""

from __future__ import annotations

import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings, allowed_levels
from .embeddings import Embedder
from .schemas import Chunk, Retrieved
from .store import IndexData

STOPWORDS = set(
    """a an and are as at be been but by can could do does for from has have how i if in into is it its
    may might must of on or our shall should so than that the their them then there these they this those
    to under was we were what when where which while who whom why will with would you your""".split()
)
TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps clause numbers like '4.2.1' as single tokens."""
    tokens = TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def _stem(tok: str) -> str:
    """Tiny suffix stripper: 'obligations' -> 'obligation', 'requires' -> 'require'."""
    if tok[0].isdigit() or len(tok) <= 4:
        return tok
    for suf in ("ies", "es", "s"):
        if tok.endswith(suf) and not tok.endswith("ss"):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


class LuceneBM25(BM25Okapi):
    """BM25 with Lucene/Elasticsearch IDF: log(1 + (N - n + 0.5) / (n + 0.5)).

    rank_bm25's Okapi IDF, log((N - n + 0.5) / (n + 0.5)), is zero or negative for any
    term found in half or more of the documents. In a small corpus (or with large
    fixed-size windows) that wipes out every score. The Lucene form is always positive.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


def index_text(chunk: Chunk) -> str:
    """What we actually index: titles help short clauses match topical queries."""
    return f"{chunk.doc_title}\n{chunk.section_id} {chunk.section_title}\n{chunk.text}"


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fuse several ranked lists of item ids into one score per id."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """search() = [rewrite query] -> BM25 (+ dense) per query -> RRF -> [rerank top pool] -> top_k."""

    def __init__(self, index: IndexData, settings: Settings, embedder: Embedder | None = None,
                 rewriter=None, reranker=None):
        self.chunks = index.chunks
        self.settings = settings
        self.embedder = embedder
        self.rewriter = rewriter  # rewrite.Rewriter | None
        self.reranker = reranker  # rerank.Reranker | None
        self.embeddings = index.embeddings
        if embedder is not None and self.embeddings is None:
            raise ValueError("Dense search requested but the index has no embeddings; re-run ingest with an embedder.")
        built_with = index.meta.get("embedder")
        if embedder is not None and built_with and built_with != embedder.name:
            raise ValueError(
                f"Index embeddings were built with {built_with!r}, but query embedder is {embedder.name!r}."
            )
        self.bm25 = LuceneBM25([tokenize(index_text(c)) for c in self.chunks]) if self.chunks else None
        self._levels = np.array([c.access_level for c in self.chunks])
        self._doc_ids = np.array([c.doc_id for c in self.chunks])

    def visible_documents(self, role: str) -> dict[str, dict]:
        """doc_id -> {title, access_level, chunks} for documents this role may see."""
        allowed = allowed_levels(role)
        docs: dict[str, dict] = {}
        for c in self.chunks:
            if c.access_level in allowed:
                d = docs.setdefault(c.doc_id, {"title": c.doc_title, "access_level": c.access_level, "chunks": 0})
                d["chunks"] += 1
        return docs

    # ---- individual retrievers -------------------------------------------------

    def _allowed_mask(self, role: str) -> np.ndarray:
        return np.isin(self._levels, list(allowed_levels(role)))

    def bm25_ranking(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        if self.bm25 is None:
            return []
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        scores[~mask] = -np.inf
        order = np.argsort(-scores)[:n]
        # A zero BM25 score means no query term matched at all: not a real hit.
        return [int(i) for i in order if scores[i] > 0]

    def dense_ranking(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        if self.embedder is None or self.embeddings is None:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.embeddings @ q
        sims[~mask] = -np.inf
        order = np.argsort(-sims)[:n]
        return [int(i) for i in order if np.isfinite(sims[i]) and sims[i] > 0]

    # ---- public API ------------------------------------------------------------

    def search(self, query: str, role: str = "public", top_k: int | None = None,
               doc_ids: set[str] | None = None) -> list[Retrieved]:
        """Top-k chunks for `query` that `role` may see (optionally only from `doc_ids`)."""
        top_k = top_k or self.settings.top_k
        if not self.chunks or not query.strip():
            return []
        mask = self._allowed_mask(role)
        if doc_ids:
            mask &= np.isin(self._doc_ids, list(doc_ids))
        pool = max(self.settings.candidate_pool, top_k)
        queries = [query] + (self.rewriter.rewrite(query) if self.rewriter else [])
        rankings: list[list[int]] = []
        bm25_pos: dict[int, int] = {}
        dense_pos: dict[int, int] = {}
        for qi, q in enumerate(queries):
            bm25 = self.bm25_ranking(q, mask, pool)
            dense = self.dense_ranking(q, mask, pool)
            rankings += [r for r in (bm25, dense) if r]
            if qi == 0:  # ranks reported for the original query (debugging/eval)
                bm25_pos = {idx: r for r, idx in enumerate(bm25, start=1)}
                dense_pos = {idx: r for r, idx in enumerate(dense, start=1)}
        fused = rrf_fuse(rankings, k=self.settings.rrf_k)
        n_first_stage = max(top_k, self.settings.rerank_pool) if self.reranker else top_k
        best = sorted(fused, key=lambda i: (-fused[i], i))[:n_first_stage]
        results = [
            Retrieved(chunk=self.chunks[i], score=fused[i], bm25_rank=bm25_pos.get(i), dense_rank=dense_pos.get(i))
            for i in best
        ]
        if self.reranker and results:
            results = self.reranker.rerank(query, results)
        return results[:top_k]

    def pop_aux_usage(self) -> list:
        """LLM calls made by the rewriter/reranker since the last call (for cost accounting)."""
        out = []
        for comp in (self.rewriter, self.reranker):
            usage = getattr(comp, "usage", None)
            if usage:
                out.extend(usage)
                usage.clear()
        return out
