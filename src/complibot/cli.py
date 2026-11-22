"""Command line interface.

    complibot ingest [--chunking section|fixed] [--embedder none|hash|openai]
    complibot inspect data/raw/internal/mas-trm-guidelines-2021.pdf   # check clause detection
    complibot search "What is the RTO for critical systems?" --role internal
    complibot ask "..." [--role public|internal] [--provider fake|anthropic|openai] [--json]
    complibot agent "Make a checklist for AI incident reporting" --role internal
    complibot serve [--port 8000]
    complibot eval [--golden eval/golden/acme_demo.jsonl] [--system rag|oracle|null] [--reps 2]
    complibot eval-generate --per-doc 15 --out eval/golden/candidates.jsonl
    complibot eval-label eval/golden/candidates.jsonl --out eval/golden/sg_regulations.jsonl
    complibot eval-calibrate [--file eval/golden/judge_calibration.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings


def _settings(args) -> Settings:
    s = Settings.from_env()
    changes = {}
    for attr in ("chunking", "embedder", "data_dir", "index_dir", "rewriter", "reranker"):
        val = getattr(args, attr, None)
        if val:
            changes[attr] = Path(val) if attr.endswith("_dir") else val
    if getattr(args, "provider", None):
        changes["llm_provider"] = args.provider
    return s.with_(**changes)


def cmd_ingest(args) -> int:
    from .embeddings import get_embedder
    from .ingest import build_index, discover_files

    s = _settings(args)
    files = discover_files(s.data_dir)
    if not files:
        print(f"No .pdf/.txt/.md files under {s.data_dir}/public or {s.data_dir}/internal", file=sys.stderr)
        return 1
    data = build_index(s, embedder=get_embedder(s))
    by_doc: dict[str, tuple[str, int]] = {}
    for c in data.chunks:
        level, n = by_doc.get(c.doc_id, (c.access_level, 0))
        by_doc[c.doc_id] = (level, n + 1)
    print(f"Indexed {len(data.chunks)} chunks from {len(by_doc)} documents -> {s.index_dir}")
    print(f"  chunking={s.chunking} embedder={data.meta['embedder'] or 'none (BM25 only)'}")
    for doc, (level, n) in sorted(by_doc.items()):
        print(f"  [{level:8}] {doc}: {n} chunks")
    return 0


def cmd_inspect(args) -> int:
    from .ingest import ingest_file, load_catalog

    s = _settings(args)
    path = Path(args.file)
    chunks = ingest_file(path, s.data_dir, strategy=s.chunking, catalog=load_catalog(s.data_dir))
    for c in chunks[: args.limit]:
        preview = " ".join(c.text.split())[:90]
        print(f"{c.section_id:>10} p{c.page_start:<4} {c.section_title[:40]:40} | {preview}")
    print(f"\n{len(chunks)} chunks ({c.access_level if chunks else '-'}). Showing {min(args.limit, len(chunks))}.")
    return 0


def cmd_search(args) -> int:
    from .answer import CompliBot

    bot = CompliBot.from_settings(_settings(args))
    for r in bot.retriever.search(args.question, role=args.role, top_k=args.top_k):
        c = r.chunk
        print(f"{r.score:.4f} bm25#{r.bm25_rank} dense#{r.dense_rank}  {c.doc_id} s.{c.section_id} p.{c.page_start}"
              f" [{c.access_level}]\n    {' '.join(c.text.split())[:160]}")
    return 0


def cmd_ask(args) -> int:
    from .answer import CompliBot

    bot = CompliBot.from_settings(_settings(args))
    result = bot.ask(args.question, role=args.role, top_k=args.top_k)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    print(f"\n{result.answer}\n")
    if result.status == "unsupported":
        print(f"WARNING: answer is not fully supported by retrieved clauses (invalid: {result.invalid_citations})")
    if result.error:
        print(f"ERROR: {result.error}")
    for c in result.citations:
        print(f"  - {c.doc_title}, s.{c.section_id} {c.section_title} (p.{c.page_start})")
    print(f"\n[status={result.status} role={result.role} latency={result.latency_ms:.0f}ms "
          f"tokens={result.usage.input_tokens}/{result.usage.output_tokens} cost=${result.usage.cost_usd:.5f}]")
    return 0


def cmd_agent(args) -> int:
    from .agent import Agent
    from .answer import CompliBot

    bot = CompliBot.from_settings(_settings(args))
    result = Agent.from_bot(bot).run(args.question, role=args.role)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    for i, step in enumerate(result.steps, start=1):
        print(f"  step {i}: {step.tool}({json.dumps(step.args)[:80]}) -> {'ok' if step.ok else 'ERROR'}: {step.summary[:90]}")
    print(f"\n{result.answer}\n")
    if result.checklist:
        print(json.dumps(result.checklist, indent=2))
    for c in result.citations:
        print(f"  - {c.doc_title}, s.{c.section_id} {c.section_title} (p.{c.page_start})")
    print(f"\n[status={result.status} runner={result.runner} role={result.role} pii={result.pii_redacted} "
          f"latency={result.latency_ms:.0f}ms cost=${result.usage.cost_usd:.5f}]")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("complibot.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cmd_eval(args) -> int:
    import time as _time

    from .answer import CompliBot
    from .evaluation import check_thresholds, format_table, load_golden, run_eval, save_run, summarize
    from .judge import get_judge

    s = _settings(args)
    if args.top_k:
        s = s.with_(top_k=args.top_k)
    items = load_golden(Path(args.golden), include_unverified=args.include_unverified)
    if not items:
        print(f"No verified items in {args.golden}", file=sys.stderr)
        return 1
    bot = CompliBot.from_settings(s)
    judge = None if args.no_judge else get_judge(s, provider=args.judge_provider)
    label = args.label or f"{args.system}-{s.chunking}-{s.embedder}-{bot.llm.name}"
    print(f"Evaluating {len(items)} items x {args.reps} rep(s): system={args.system} llm={bot.llm.name}:"
          f"{bot.llm.model} judge={judge.name if judge else 'off'}")
    rows = run_eval(items, bot, judge, system=args.system, reps=args.reps, progress=sys.stdout.isatty())
    summary = summarize(rows)
    table = format_table(summary, title=f"CompliBot eval: {label}")
    config = {"label": label, "golden": args.golden, "system": args.system, "reps": args.reps,
              "llm": f"{bot.llm.name}:{bot.llm.model}", "judge": judge.name if judge else None,
              "chunking": s.chunking, "embedder": s.embedder, "top_k": s.top_k, "git": _git_sha()}
    out = save_run(Path(args.out_dir) / f"{_time.strftime('%Y%m%d-%H%M%S')}-{label}", rows, summary, config, table)
    print(table)
    print(f"\nSaved to {out}")
    mins = dict(m.split("=", 1) for m in args.min or [])
    failures = check_thresholds(summary, {k: float(v) for k, v in mins.items()})
    for f in failures:
        print(f"THRESHOLD FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


def cmd_eval_generate(args) -> int:
    from .labeling import append_jsonl, fake_generator_provider, generate_candidates
    from .llm import get_provider
    from .store import load_index

    s = _settings(args)
    chunks = load_index(s.index_dir).chunks
    llm = fake_generator_provider() if s.llm_provider == "fake" else get_provider(s)
    items = generate_candidates(chunks, llm, per_doc=args.per_doc, seed=args.seed)
    append_jsonl(Path(args.out), items)
    print(f"Wrote {len(items)} unverified candidates to {args.out}. Next: complibot eval-label {args.out} --out ...")
    return 0


def cmd_eval_label(args) -> int:
    from .answer import CompliBot
    from .labeling import HELP, append_jsonl, read_jsonl, label_items

    bot = CompliBot.from_settings(_settings(args))
    items = read_jsonl(Path(args.file))
    print(HELP)
    done = label_items(items, bot.retriever)
    append_jsonl(Path(args.out), done)
    print(f"\nSaved {len(done)} verified items to {args.out}")
    return 0


def cmd_eval_calibrate(args) -> int:
    from .judge import get_judge
    from .labeling import read_jsonl

    s = _settings(args)
    judge = get_judge(s, provider=args.judge_provider)
    rows = read_jsonl(Path(args.file))
    agree, tp, fp, fn, tn = 0, 0, 0, 0, 0
    for r in rows:
        res = judge.faithfulness(r["question"], r["answer"], [tuple(x) for x in r["sources"]])
        pred = res.score is not None and res.score >= 0.999
        human = bool(r["human_faithful"])
        agree += pred == human
        tp += pred and human
        fp += pred and not human
        fn += (not pred) and human
        tn += (not pred) and (not human)
        if pred != human:
            print(f"  disagree {r['id']}: human={human} judge={res.score} {res.detail.get('claims', '')}")
    n = len(rows)
    print(f"Judge {judge.name}: agreement {agree}/{n} = {agree / n:.0%}  (TP={tp} FP={fp} FN={fn} TN={tn})")
    print("Aim for >= 90% on clear-cut cases before trusting faithfulness scores.")
    return 0


def cmd_experiment(args) -> int:
    from .experiments import run_experiments, update_readme

    config = json.loads(Path(args.config).read_text())
    out = Path(args.out_dir) / Path(args.config).stem
    report = run_experiments(config, out, base_settings=_settings(args), progress=True)
    md = (out / "comparison.md").read_text()
    print("\n" + md)
    print(f"Saved {out}/comparison.md and comparison.json ({len(report['experiments'])} configurations)")
    if args.update_readme:
        ok = update_readme(Path("README.md"), md)
        print("README.md updated" if ok else "README.md has no experiments markers; not updated")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="complibot", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--index-dir", dest="index_dir")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="Build the index from data/raw")
    sp.add_argument("--chunking", choices=["section", "fixed"])
    sp.add_argument("--embedder", choices=["none", "hash", "openai"])
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("inspect", help="Show the clauses detected in one file")
    sp.add_argument("file")
    sp.add_argument("--chunking", choices=["section", "fixed"])
    sp.add_argument("--limit", type=int, default=60)
    sp.set_defaults(func=cmd_inspect)

    for name, func in (("search", cmd_search), ("ask", cmd_ask), ("agent", cmd_agent)):
        sp = sub.add_parser(name, help=f"{name} the index")
        sp.add_argument("question")
        sp.add_argument("--role", default="public", choices=["public", "internal"])
        sp.add_argument("--top-k", dest="top_k", type=int)
        sp.add_argument("--embedder", choices=["none", "hash", "openai"])
        sp.add_argument("--provider", choices=["fake", "anthropic", "openai"])
        sp.add_argument("--rewriter", choices=["none", "rules", "llm"])
        sp.add_argument("--reranker", choices=["none", "lexical", "cross-encoder", "llm"])
        if name in ("ask", "agent"):
            sp.add_argument("--json", action="store_true", help="Print the full result as JSON")
        sp.set_defaults(func=func)

    sp = sub.add_parser("eval", help="Run the golden-set evaluation and print a results table")
    sp.add_argument("--golden", default="eval/golden/acme_demo.jsonl")
    sp.add_argument("--system", choices=["rag", "oracle", "null"], default="rag",
                    help="oracle/null are sanity baselines for the harness itself")
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--provider", choices=["fake", "anthropic", "openai"])
    sp.add_argument("--judge-provider", dest="judge_provider", choices=["fake", "anthropic", "openai"])
    sp.add_argument("--no-judge", dest="no_judge", action="store_true")
    sp.add_argument("--chunking", choices=["section", "fixed"])
    sp.add_argument("--embedder", choices=["none", "hash", "openai"])
    sp.add_argument("--top-k", dest="top_k", type=int)
    sp.add_argument("--rewriter", choices=["none", "rules", "llm"])
    sp.add_argument("--reranker", choices=["none", "lexical", "cross-encoder", "llm"])
    sp.add_argument("--label")
    sp.add_argument("--out-dir", dest="out_dir", default="eval/results/runs")
    sp.add_argument("--include-unverified", dest="include_unverified", action="store_true")
    sp.add_argument("--min", action="append", metavar="METRIC=VALUE",
                    help="fail (exit 1) if a metric mean is below VALUE, e.g. --min recall@5=0.8")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("experiment", help="Run A/B experiments and write a markdown comparison table")
    sp.add_argument("--config", default="eval/experiments/offline.json")
    sp.add_argument("--out-dir", dest="out_dir", default="eval/results/experiments")
    sp.add_argument("--update-readme", dest="update_readme", action="store_true")
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("eval-generate", help="LLM-generate candidate golden items from the index")
    sp.add_argument("--per-doc", dest="per_doc", type=int, default=15)
    sp.add_argument("--seed", type=int, default=13)
    sp.add_argument("--out", default="eval/golden/candidates.jsonl")
    sp.add_argument("--provider", choices=["fake", "anthropic", "openai"])
    sp.set_defaults(func=cmd_eval_generate)

    sp = sub.add_parser("eval-label", help="Interactively verify/label golden items")
    sp.add_argument("file")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_eval_label)

    sp = sub.add_parser("eval-calibrate", help="Measure judge agreement with human faithfulness labels")
    sp.add_argument("--file", default="eval/golden/judge_calibration.jsonl")
    sp.add_argument("--judge-provider", dest="judge_provider", choices=["fake", "anthropic", "openai"])
    sp.set_defaults(func=cmd_eval_calibrate)

    sp = sub.add_parser("serve", help="Run the FastAPI server")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
