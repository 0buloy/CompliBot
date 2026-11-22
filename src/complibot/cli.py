"""Command line interface.

    complibot ingest [--chunking section|fixed] [--embedder none|hash|openai]
    complibot inspect data/raw/internal/mas-trm-guidelines-2021.pdf   # check clause detection
    complibot search "What is the RTO for critical systems?" --role internal
    complibot ask "..." [--role public|internal] [--provider fake|anthropic|openai] [--json]
    complibot serve [--port 8000]
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
    for attr in ("chunking", "embedder", "data_dir", "index_dir"):
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


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("complibot.api:app", host=args.host, port=args.port, reload=args.reload)
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

    for name, func in (("search", cmd_search), ("ask", cmd_ask)):
        sp = sub.add_parser(name, help=f"{name} the index")
        sp.add_argument("question")
        sp.add_argument("--role", default="public", choices=["public", "internal"])
        sp.add_argument("--top-k", dest="top_k", type=int)
        sp.add_argument("--embedder", choices=["none", "hash", "openai"])
        sp.add_argument("--provider", choices=["fake", "anthropic", "openai"])
        if name == "ask":
            sp.add_argument("--json", action="store_true", help="Print the full AnswerResult as JSON")
        sp.set_defaults(func=func)

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
