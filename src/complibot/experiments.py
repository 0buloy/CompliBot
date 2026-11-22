"""A/B experiments: run the golden set under several configurations and compare.

    complibot experiment --config eval/experiments/offline.json [--update-readme]

Config (JSON):
    {"golden": "eval/golden/acme_demo.jsonl", "reps": 1, "judge_provider": "fake",
     "baseline": {"chunking": "section", "reranker": "none", ...},   # Settings fields
     "experiments": [{"name": "Fixed-size chunks", "changes": {"chunking": "fixed"}}, ...]}

Each experiment = baseline settings + its changes. Comparisons against the
baseline are PAIRED (same question, same rep), which removes question-difficulty
variance and makes small differences detectable with fewer items. A difference
is only called significant when its 95% CI excludes zero.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

from .answer import CompliBot
from .config import Settings
from .embeddings import get_embedder
from .evaluation import load_golden, run_eval, save_run, summarize, format_table
from .ingest import build_index
from .judge import get_judge

SETTING_ALIASES = {"provider": "llm_provider"}
PAIRED_METRICS = ("recall@1", "recall@5", "mrr", "refusal_correct", "faithfulness", "correctness")
README_START, README_END = "<!-- experiments:start -->", "<!-- experiments:end -->"


def apply_changes(base: Settings, changes: dict) -> Settings:
    fields = {SETTING_ALIASES.get(k, k): v for k, v in changes.items()}
    unknown = [k for k in fields if k not in Settings.__dataclass_fields__]
    if unknown:
        raise ValueError(f"Unknown setting(s) in experiment config: {unknown}")
    return base.with_(**fields)


def paired_delta(rows_a: list[dict], rows_b: list[dict], metric: str) -> dict | None:
    """Mean of (b - a) over cases scored in both runs, with a 95% CI."""
    a = {(r["id"], r["rep"]): r.get(metric) for r in rows_a if r["status"] != "error"}
    diffs = []
    for r in rows_b:
        key = (r["id"], r["rep"])
        if r["status"] != "error" and r.get(metric) is not None and a.get(key) is not None:
            diffs.append(float(r[metric]) - float(a[key]))
    if not diffs:
        return None
    mean = sum(diffs) / len(diffs)
    ci = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
    significant = ci is not None and ci > 0 and abs(mean) > ci
    return {"delta": mean, "ci95": ci, "n": len(diffs), "significant": significant}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def describe(s: Settings, llm_name: str) -> dict:
    retrieval = "BM25" + (f"+dense({s.embedder})" if s.embedder != "none" else "")
    if s.rewriter != "none":
        retrieval += f" +rewrite({s.rewriter})"
    if s.reranker != "none":
        retrieval += f" +rerank({s.reranker})"
    return {"chunking": s.chunking, "retrieval": retrieval, "llm": llm_name}


def run_experiments(config: dict, out_root: Path, base_settings: Settings | None = None, progress: bool = False) -> dict:
    base = apply_changes(base_settings or Settings.from_env(), config.get("baseline", {}))
    items = load_golden(Path(config["golden"]))
    reps = int(config.get("reps", 1))
    out_root.mkdir(parents=True, exist_ok=True)
    index_cache: dict[tuple, Path] = {}
    results = []
    experiments = [{"name": "Baseline", "changes": {}}] + [e for e in config["experiments"] if e.get("enabled", True)]
    for exp in experiments:
        s = apply_changes(base, exp.get("changes", {}))
        key = (s.chunking, s.embedder, s.fixed_chunk_chars, s.fixed_overlap_chars, s.max_chunk_chars)
        if key not in index_cache:  # build each distinct index once
            idx_dir = out_root / "indexes" / "-".join(map(str, key))
            build_index(s.with_(index_dir=idx_dir), embedder=get_embedder(s))
            index_cache[key] = idx_dir
        s = s.with_(index_dir=index_cache[key])
        bot = CompliBot.from_settings(s)
        judge = get_judge(s, provider=config.get("judge_provider")) if config.get("judge", True) else None
        if progress:
            print(f"- {exp['name']}: {bot.llm.name}:{bot.llm.model}")
        t0 = time.perf_counter()
        rows = run_eval(items, bot, judge, reps=reps)
        summary = summarize(rows)
        desc = describe(s, f"{bot.llm.name}:{bot.llm.model}")
        save_run(out_root / "runs" / _slug(exp["name"]), rows, summary, {**desc, "name": exp["name"], "reps": reps},
                 format_table(summary, title=exp["name"]))
        results.append({"name": exp["name"], **desc, "summary": summary, "rows": rows,
                        "wall_s": time.perf_counter() - t0})

    baseline_rows = results[0]["rows"]
    for r in results[1:]:
        r["paired"] = {m: paired_delta(baseline_rows, r["rows"], m) for m in PAIRED_METRICS}
    report = {"golden": config["golden"], "n_items": len(items), "reps": reps,
              "generated_at": time.strftime("%Y-%m-%d %H:%M"),
              "experiments": [{k: v for k, v in r.items() if k != "rows"} for r in results]}
    (out_root / "comparison.json").write_text(json.dumps(report, indent=2))
    md = comparison_markdown(report)
    (out_root / "comparison.md").write_text(md)
    return report


def _m(summary: dict, key: str) -> str:
    v = summary.get(key, {})
    return "n/a" if not v or v.get("mean") is None else f"{v['mean']:.2f}"


def _delta(p: dict | None) -> str:
    if not p:
        return "n/a"
    ci = f" ±{p['ci95']:.2f}" if p.get("ci95") else ""
    star = " *" if p["significant"] else ""
    return f"{p['delta']:+.2f}{ci}{star}"


def comparison_markdown(report: dict) -> str:
    lines = [
        f"Golden set: `{report['golden']}` ({report['n_items']} items x {report['reps']} rep). "
        f"Generated {report['generated_at']}.",
        "",
        "| Experiment | Chunking | Retrieval | LLM | Recall@1 | Recall@5 | MRR | Refusal acc. | Unsafe ans. "
        "| Faithful. | Correct. | p50 ms | $/q |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in report["experiments"]:
        s = e["summary"]
        p50 = s["latency_ms"]["p50"]
        per_q = s["cost_usd"]["answer_per_q"] or 0.0
        lines.append(
            f"| {e['name']} | {e['chunking']} | {e['retrieval']} | {e['llm']} | {_m(s, 'recall@1')} | "
            f"{_m(s, 'recall@5')} | {_m(s, 'mrr')} | {_m(s, 'refusal_accuracy')} | {_m(s, 'unsafe_answer_rate')} | "
            f"{_m(s, 'faithfulness')} | {_m(s, 'correctness')} | {0 if p50 is None else p50:.0f} | {per_q:.4f} |"
        )
    lines += ["", "Paired difference vs baseline (same questions; ±95% CI; * = CI excludes 0):", "",
              "| Experiment | Δ Recall@1 | Δ MRR | Δ Refusal acc. | Δ Correctness |", "|---|---|---|---|---|"]
    for e in report["experiments"][1:]:
        p = e["paired"]
        lines.append(f"| {e['name']} | {_delta(p['recall@1'])} | {_delta(p['mrr'])} | "
                     f"{_delta(p['refusal_correct'])} | {_delta(p['correctness'])} |")
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, markdown: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    readme.write_text(f"{pre}{README_START}\n{markdown}{README_END}{post}", encoding="utf-8")
    return True
