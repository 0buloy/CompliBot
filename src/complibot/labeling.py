"""Tools for writing the golden set (target: 80-150 items).

Method (see eval/README.md for the full workflow):

1. Generate candidates with an LLM, one clause at a time, so the source clause
   is known by construction:
       complibot eval-generate --per-doc 15 --out eval/golden/candidates.jsonl
   It also writes "near-miss" unanswerable questions (related topic, not covered).
2. Label/verify every candidate (and every hand-written draft) interactively:
       complibot eval-label eval/golden/candidates.jsonl --out eval/golden/sg_regulations.jsonl
   For each item you see the top retrieved clauses and confirm which ones really
   answer it (or mark it unanswerable), fix the question, and accept or rewrite
   the reference answer. Only labelled items get "verified": true.
3. Hand-write the hard ones the generator can't: multi-clause comparisons,
   questions using different words from the document, role-gated questions
   (internal content asked as "public"), and out-of-scope questions.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .llm import FakeProvider, LLMError, LLMProvider, parse_json_object
from .retrieve import HybridRetriever
from .schemas import Chunk

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
        "near_miss_question": {"type": "string"},
    },
    "required": ["question", "reference_answer", "near_miss_question"],
    "additionalProperties": False,
}

GEN_SYSTEM = """You write evaluation questions for a Singapore regulatory question-answering system.
You receive ONE clause from a regulatory document. Write:
1. "question": a realistic question a compliance officer might ask that this clause answers. Do not
   copy phrases verbatim from the clause; paraphrase as a real user would. Do not mention the clause
   number.
2. "reference_answer": a 1-2 sentence answer using only this clause.
3. "near_miss_question": a related, realistic question on the same topic that this clause does NOT
   answer (e.g. asks for a number, deadline, penalty, or scope the clause does not state).
Return JSON with exactly these three keys."""


def _fake_generator(system: str, user: str, schema: dict | None) -> str:
    title = re.search(r'title="([^"]*)"', user)
    topic = title.group(1).lower() if title else "this topic"
    body = re.sub(r"<[^>]+>", "", user).strip().split(". ")[0]
    return json.dumps({
        "question": f"What does the guidance say about {topic}?",
        "reference_answer": body[:300],
        "near_miss_question": f"What is the penalty for not complying with the rules on {topic}?",
    })


def fake_generator_provider() -> FakeProvider:
    return FakeProvider(responder=_fake_generator)


def sample_chunks(chunks: list[Chunk], per_doc: int, seed: int = 13, min_chars: int = 150) -> list[Chunk]:
    """Spread samples evenly across each document (stratified), skipping tiny chunks."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text) >= min_chars and c.section_id != "preamble":
            by_doc[c.doc_id].append(c)
    rng = random.Random(seed)
    out = []
    for doc in sorted(by_doc):
        pool = by_doc[doc]
        if len(pool) <= per_doc:
            out.extend(pool)
            continue
        step = len(pool) / per_doc  # one random clause from each of per_doc even slices
        out.extend(pool[int(i * step) + rng.randrange(max(1, int(step)))] for i in range(per_doc))
    return out


def generate_candidates(chunks: list[Chunk], llm: LLMProvider, per_doc: int = 15, seed: int = 13) -> list[dict]:
    items = []
    for c in sample_chunks(chunks, per_doc, seed):
        user = f'<clause doc="{c.doc_title}" section="{c.section_id}" title="{c.section_title}">\n{c.text}\n</clause>'
        try:
            data = parse_json_object(llm.complete(GEN_SYSTEM, user, json_schema=GEN_SCHEMA).text)
        except (LLMError, ValueError) as e:
            print(f"  skip {c.chunk_id}: {e}")
            continue
        role = "public" if c.access_level == "public" else "internal"
        base = re.sub(r"[^a-z0-9]+", "-", c.chunk_id.lower()).strip("-")
        items.append({
            "id": f"gen-{base}", "question": data["question"], "role": role, "answerable": True,
            "source_clauses": [{"doc_id": c.doc_id, "section_id": c.section_id}],
            "reference_answer": data["reference_answer"], "tags": ["generated", c.doc_id], "verified": False,
        })
        items.append({
            "id": f"gen-{base}-nearmiss", "question": data["near_miss_question"], "role": role, "answerable": False,
            "source_clauses": [], "reference_answer": "", "tags": ["near-miss", c.doc_id], "verified": False,
            "note": f"near miss of {c.chunk_id}: check no OTHER clause answers it before marking unanswerable",
        })
    return items


# -------------------------------------------------------------------- labelling

HELP = """  numbers (e.g. 1,3)  = these clauses answer it      u = unanswerable for this role
  s = skip   q = quit (progress is saved)   e = edit question   r = switch role public/internal"""


def label_items(
    items: list[dict], retriever: HybridRetriever, ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print, top_k: int = 8,
) -> list[dict]:
    """Interactive loop; returns the items that were labelled (verified=True)."""
    done = []
    for n, item in enumerate(items, start=1):
        if item.get("verified"):
            done.append(item)
            continue
        while True:
            role = item.get("role", "public")
            hits = retriever.search(item["question"], role=role, top_k=top_k)
            show(f"\n[{n}/{len(items)}] {item['id']}  role={role}  draft answerable={item.get('answerable')}")
            show(f"Q: {item['question']}")
            if item.get("source_clauses"):
                show(f"   draft source: {item['source_clauses']}")
            if item.get("note"):
                show(f"   note: {item['note']}")
            for i, h in enumerate(hits, start=1):
                c = h.chunk
                show(f"  {i}. {c.doc_id} s.{c.section_id} p.{c.page_start} - {' '.join(c.text.split())[:150]}")
            choice = ask("label> ").strip().lower()
            if choice == "q":
                return done
            if choice == "s":
                break
            if choice == "e":
                item["question"] = ask("new question> ").strip() or item["question"]
                continue
            if choice == "r":
                item["role"] = "internal" if role == "public" else "public"
                continue
            if choice == "u":
                item.update(answerable=False, source_clauses=[], reference_answer="", verified=True)
                done.append(item)
                break
            picks = [int(x) for x in re.findall(r"\d+", choice) if 1 <= int(x) <= len(hits)]
            if not picks:
                show(HELP)
                continue
            clauses = [{"doc_id": hits[i - 1].chunk.doc_id, "section_id": hits[i - 1].chunk.section_id} for i in picks]
            item.update(answerable=True, source_clauses=clauses)
            default = item.get("reference_answer") or ""
            show(f"   reference answer: {default or '(none)'}")
            new = ask("reference answer (enter = keep)> ").strip()
            item["reference_answer"] = new or default
            item["verified"] = True
            item.pop("note", None)
            done.append(item)
            break
    return done


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]


def append_jsonl(path: Path, items: list[dict]) -> None:
    """Append items, replacing any existing item with the same id."""
    existing = {i["id"]: i for i in read_jsonl(path)} if path.exists() else {}
    for item in items:
        existing[item["id"]] = item
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in existing.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
