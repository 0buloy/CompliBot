"""Evaluation harness: golden set -> per-case rows -> summary metrics -> table.

Golden item (one JSON object per line):
    {
      "id": "acme-007",
      "question": "Who must review AI-generated content before it is sent to customers?",
      "role": "internal",                 # role the question is asked as
      "answerable": true,                 # answerable FOR THAT ROLE from the corpus
      "source_clauses": [{"doc_id": "acme-ai-usage-policy", "section_id": "4.1"}],
      "reference_answer": "A human employee must review and approve it first.",
      "tags": ["human-oversight"],        # first tag = category used for breakdowns
      "verified": true                    # a human checked question, clauses and answer
    }

Metrics (all recomputable from results.jsonl):
- retrieval: recall@k (share of gold clauses in the top k), hit@k, MRR. Answerable items only.
- refusal: accuracy, false-refusal rate (refused an answerable q), unsafe-answer rate
  (answered an unanswerable q).
- grounding: supported rate (all citations valid), citation precision (cited chunks
  that cover a gold clause), faithfulness (LLM judge, claims supported by cited text).
- correctness: LLM judge vs reference answer (yes=1, partial=0.5, no=0).
- ops: latency p50/p95 (end-to-end and LLM-only), tokens, cost (answer and judge separately).

Infra failures (status "error": provider down, unparseable output) go to errors.jsonl and are
excluded from quality metrics, so plumbing problems are never scored as model mistakes.

Sanity baselines: --system oracle (gold clauses + reference answers; should score ~100%) and
--system null (always "I don't know"; should fail every answerable item). If they don't, the
harness or grader is broken.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .answer import IDK, CompliBot, answer_question
from .judge import Judge
from .schemas import AnswerResult, Citation, Retrieved

KS = (1, 3, 5, 10)


# ----------------------------------------------------------------- golden set


@dataclass
class GoldenItem:
    id: str
    question: str
    answerable: bool
    role: str = "public"
    source_clauses: list[dict] = field(default_factory=list)
    reference_answer: str = ""
    tags: list[str] = field(default_factory=list)
    verified: bool = False

    @property
    def gold(self) -> set[tuple[str, str]]:
        return {(c["doc_id"], str(c["section_id"])) for c in self.source_clauses}

    @property
    def category(self) -> str:
        return self.tags[0] if self.tags else "untagged"


def load_golden(path: Path, include_unverified: bool = False) -> list[GoldenItem]:
    items, seen = [], set()
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            raw = json.loads(line)
            known = {k: raw[k] for k in GoldenItem.__dataclass_fields__ if k in raw}
            item = GoldenItem(**known)
            if item.id in seen:
                raise ValueError(f"{path}:{n}: duplicate id {item.id}")
            if item.verified and item.answerable and not item.source_clauses:
                raise ValueError(f"{path}:{n}: answerable item {item.id} has no source_clauses")
            seen.add(item.id)
            if item.verified or include_unverified:
                items.append(item)
    return items


# -------------------------------------------------------------- per-case scoring


def chunk_keys(r: Retrieved) -> set[tuple[str, str]]:
    return {(r.chunk.doc_id, sec) for sec in (r.chunk.covers or [r.chunk.section_id])}


def retrieval_metrics(gold: set[tuple[str, str]], retrieved: list[Retrieved]) -> dict:
    out: dict = {}
    if not gold:
        return out
    for k in KS:
        found = set().union(*(chunk_keys(r) for r in retrieved[:k])) if retrieved[:k] else set()
        out[f"recall@{k}"] = len(gold & found) / len(gold)
        out[f"hit@{k}"] = 1.0 if gold & found else 0.0
    rr = 0.0
    for rank, r in enumerate(retrieved, start=1):
        if gold & chunk_keys(r):
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def citation_precision(gold: set[tuple[str, str]], citations: list[Citation], by_id: dict) -> float | None:
    if not citations:
        return None
    good = sum(1 for c in citations if c.chunk_id in by_id and gold & chunk_keys(by_id[c.chunk_id]))
    return good / len(citations)


# -------------------------------------------------------------- systems under test


def _oracle(item: GoldenItem, bot: CompliBot, retrieved: list[Retrieved]) -> tuple[list[Retrieved], AnswerResult]:
    """Perfect system: gold clauses ranked first, reference answer, gold citations."""
    gold_hits = [r for r in retrieved if item.gold & chunk_keys(r)]
    if item.answerable:  # make sure every gold clause is present even if search missed it
        have = set().union(*(chunk_keys(r) for r in gold_hits)) if gold_hits else set()
        for c in bot.retriever.chunks:
            key = (c.doc_id, c.section_id)
            if key in item.gold and key not in have:
                gold_hits.append(Retrieved(chunk=c, score=1.0))
                have.add(key)
    ranked = gold_hits + [r for r in retrieved if r not in gold_hits]
    if not item.answerable:
        res = AnswerResult(question=item.question, answer=IDK, status="refused", refused=True, supported=True,
                           role=item.role)
        return ranked, res
    from .answer import _to_citation

    cites = [_to_citation(r) for r in gold_hits]
    res = AnswerResult(question=item.question, answer=item.reference_answer, status="answered", refused=False,
                       supported=True, citations=cites, role=item.role,
                       retrieved_ids=[r.chunk.chunk_id for r in ranked])
    return ranked, res


def _null(item: GoldenItem, bot: CompliBot, retrieved: list[Retrieved]) -> tuple[list[Retrieved], AnswerResult]:
    return [], AnswerResult(question=item.question, answer=IDK, status="refused", refused=True, supported=True,
                            role=item.role)


def run_case(item: GoldenItem, bot: CompliBot, judge: Judge | None, system: str = "rag", rep: int = 0) -> dict:
    t0 = time.perf_counter()
    bot.retriever.pop_aux_usage()
    retrieved = bot.retriever.search(item.question, role=item.role, top_k=max(KS))
    retrieval_ms = (time.perf_counter() - t0) * 1000
    aux = bot.retriever.pop_aux_usage()  # LLM calls made by query rewriting / reranking
    if system == "oracle":
        retrieved, result = _oracle(item, bot, retrieved)
    elif system == "null":
        retrieved, result = _null(item, bot, retrieved)
    else:
        top_k = bot.settings.top_k
        result = answer_question(item.question, bot.retriever, bot.llm, role=item.role, retrieved=retrieved[:top_k])
    by_id = {r.chunk.chunk_id: r for r in retrieved}

    row: dict = {
        "id": item.id, "rep": rep, "system": system, "question": item.question, "role": item.role,
        "category": item.category, "answerable": item.answerable,
        "gold": sorted(f"{d}::{s}" for d, s in item.gold),
        "retrieved": [r.chunk.chunk_id for r in retrieved],
        "status": result.status, "refused": result.refused, "supported": result.supported,
        "answer": result.answer, "citations": [c.chunk_id for c in result.citations],
        "invalid_citations": result.invalid_citations, "error": result.error,
        "latency_ms": retrieval_ms + result.latency_ms, "llm_latency_ms": result.usage.llm_latency_ms,
        "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
        "cost_usd": result.usage.cost_usd + sum(a.cost_usd for a in aux),
        "aux_llm_calls": len(aux), "served_model": result.usage.model,
    }
    row["input_tokens"] += sum(a.input_tokens for a in aux)
    row["output_tokens"] += sum(a.output_tokens for a in aux)
    if result.status == "error":
        return row

    row.update(retrieval_metrics(item.gold, retrieved) if item.answerable else {})
    row["refusal_correct"] = result.refused == (not item.answerable)
    if item.answerable and not result.refused:
        row["citation_precision"] = citation_precision(item.gold, result.citations, by_id)
        if judge is not None:
            sources = [(c.chunk_id, by_id[c.chunk_id].chunk.text) for c in result.citations if c.chunk_id in by_id]
            f = judge.faithfulness(item.question, result.answer, sources)
            row.update(faithfulness=f.score, faithfulness_detail=f.detail, judge_error=f.error)
            judge_cost, judge_tokens = f.cost_usd, f.input_tokens + f.output_tokens
            if item.reference_answer:
                c = judge.correctness(item.question, result.answer, item.reference_answer)
                row.update(correctness=c.score, correctness_detail=c.detail)
                row["judge_error"] = row["judge_error"] or c.error
                judge_cost, judge_tokens = judge_cost + c.cost_usd, judge_tokens + c.input_tokens + c.output_tokens
            row.update(judge_cost_usd=judge_cost, judge_tokens=judge_tokens)
    return row


# --------------------------------------------------------------------- summary


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _ci95(xs: list[float]) -> float | None:
    """Half-width of a normal-approximation 95% CI for the mean."""
    if len(xs) < 2:
        return None
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] != "error"]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]

    def metric(name: str, subset: list[dict]) -> dict:
        xs = [float(r[name]) for r in subset if r.get(name) is not None]
        return {"mean": _mean(xs), "ci95": _ci95(xs), "n": len(xs)}

    s: dict = {"n_rows": len(rows), "n_errors": len(rows) - len(ok), "n_answerable": len(ans),
               "n_unanswerable": len(unans)}
    for k in KS:
        s[f"recall@{k}"] = metric(f"recall@{k}", ans)
    s["mrr"] = metric("mrr", ans)
    s["refusal_accuracy"] = metric("refusal_correct", ok)
    s["false_refusal_rate"] = {"mean": _mean([1.0 if r["refused"] else 0.0 for r in ans]), "n": len(ans)}
    s["unsafe_answer_rate"] = {"mean": _mean([0.0 if r["refused"] else 1.0 for r in unans]), "n": len(unans)}
    answered = [r for r in ans if not r["refused"]]
    s["supported_rate"] = {"mean": _mean([1.0 if r["supported"] else 0.0 for r in answered]), "n": len(answered)}
    s["citation_precision"] = metric("citation_precision", answered)
    s["faithfulness"] = metric("faithfulness", answered)
    s["correctness"] = metric("correctness", answered)
    lat = [r["latency_ms"] for r in ok]
    llm_lat = [r["llm_latency_ms"] for r in ok if r["llm_latency_ms"]]
    s["latency_ms"] = {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95)}
    s["llm_latency_ms"] = {"p50": _pct(llm_lat, 0.5), "p95": _pct(llm_lat, 0.95)}
    s["tokens"] = {"input": sum(r["input_tokens"] for r in rows), "output": sum(r["output_tokens"] for r in rows)}
    s["cost_usd"] = {"answer_total": sum(r["cost_usd"] for r in rows),
                     "answer_per_q": _mean([r["cost_usd"] for r in rows]),
                     "judge_total": sum(r.get("judge_cost_usd", 0.0) for r in rows)}
    s["judge_errors"] = sum(1 for r in ok if r.get("judge_error"))
    s["served_models"] = sorted({r["served_model"] for r in rows if r["served_model"]})
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in ok}):
        sub = [r for r in ok if r["category"] == cat]
        by_cat[cat] = {"n": len(sub), "refusal_accuracy": metric("refusal_correct", sub)["mean"],
                       "recall@5": metric("recall@5", [r for r in sub if r["answerable"]])["mean"]}
    s["by_category"] = by_cat
    return s


# ---------------------------------------------------------------------- report

TABLE_ROWS = [
    ("Retrieval", "recall@1", "recall@1"), ("", "recall@3", "recall@3"), ("", "recall@5", "recall@5"),
    ("", "recall@10", "recall@10"), ("", "MRR", "mrr"),
    ("Refusal", "refusal accuracy", "refusal_accuracy"),
    ("", "false refusal rate (lower=better)", "false_refusal_rate"),
    ("", "unsafe answer rate (lower=better)", "unsafe_answer_rate"),
    ("Grounding", "supported (valid citations)", "supported_rate"), ("", "citation precision", "citation_precision"),
    ("", "faithfulness (judge)", "faithfulness"), ("Answer", "correctness (judge)", "correctness"),
]


def _fmt(m: dict | None) -> str:
    if not m or m.get("mean") is None:
        return "n/a"
    ci = m.get("ci95")
    return f"{m['mean']:.3f}" + (f" ± {ci:.3f}" if ci else "") + f"  (n={m.get('n', '?')})"


def format_table(summary: dict, title: str = "CompliBot eval") -> str:
    lines = [title, "=" * len(title)]
    w = 36
    for group, label, key in TABLE_ROWS:
        lines.append(f"{group:10} {label:{w}} {_fmt(summary.get(key))}")
    lat, llm = summary["latency_ms"], summary["llm_latency_ms"]
    f = lambda v: "n/a" if v is None else f"{v:.0f}"  # noqa: E731
    lines.append(f"{'Ops':10} {'latency p50 / p95 ms (end-to-end)':{w}} {f(lat['p50'])} / {f(lat['p95'])}")
    lines.append(f"{'':10} {'LLM latency p50 / p95 ms':{w}} {f(llm['p50'])} / {f(llm['p95'])}")
    t, c = summary["tokens"], summary["cost_usd"]
    lines.append(f"{'':10} {'tokens in / out':{w}} {t['input']} / {t['output']}")
    per_q = c["answer_per_q"] or 0.0
    lines.append(f"{'':10} {'cost answer total (per q) USD':{w}} {c['answer_total']:.4f} ({per_q:.5f})")
    lines.append(f"{'':10} {'cost judge total USD':{w}} {c['judge_total']:.4f}")
    lines.append(f"rows={summary['n_rows']} answerable={summary['n_answerable']} "
                 f"unanswerable={summary['n_unanswerable']} errors={summary['n_errors']} "
                 f"judge_errors={summary['judge_errors']} models={','.join(summary['served_models']) or '-'}")
    return "\n".join(lines)


def check_thresholds(summary: dict, mins: dict[str, float]) -> list[str]:
    """Return failures for --min metric=value gates (used by CI)."""
    failures = []
    for name, floor in mins.items():
        m = summary.get(name)
        value = m.get("mean") if isinstance(m, dict) else None
        if value is None or value < floor:
            failures.append(f"{name}={value} < {floor}")
    return failures


# ------------------------------------------------------------------------ runner


def run_eval(items: list[GoldenItem], bot: CompliBot, judge: Judge | None, system: str = "rag",
             reps: int = 1, progress: bool = False) -> list[dict]:
    rows = []
    for rep in range(reps):
        for n, item in enumerate(items, start=1):
            rows.append(run_case(item, bot, judge, system=system, rep=rep))
            if progress:
                print(f"\r  rep {rep + 1}/{reps} case {n}/{len(items)}", end="", flush=True)
    if progress:
        print()
    return rows


def save_run(out_dir: Path, rows: list[dict], summary: dict, config: dict, table: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if r["status"] != "error":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            if r["status"] == "error":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2))
    (out_dir / "summary.txt").write_text(table + "\n")
    return out_dir
