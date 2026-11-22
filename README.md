# CompliBot

An **evaluated** RAG assistant that answers compliance questions over Singapore regulatory
documents (MAS Technology Risk Management Guidelines, PDPC PDPA advisory guidelines, IMDA Model AI
Governance Framework), with clause-level citations on every answer and an "I don't know" when the
documents don't support one.

> Status: Phases 1-4 are done (core RAG, evaluation, experiments, agent and guardrails). Deployment is next.

## What it does

- **Clause-aware ingestion.** PDFs are split on numbered headings/clauses (`4.2.1 ...`). Every chunk
  keeps doc title, clause ID, heading title, page range and access level.
- **Hybrid retrieval.** BM25 (`rank_bm25`) plus optional dense embeddings (numpy), merged with
  reciprocal rank fusion. With dense search off, it runs fully offline on BM25 alone.
- **Permission-aware retrieval.** `data/raw/public` vs `data/raw/internal` decides a document's
  access level. The role filter runs *before* ranking, so internal text never reaches a public
  user's prompt. The API works out the role from the `X-API-Key` header, never from the request body.
- **Grounded answers.** The LLM must return `{answer, citations, refused}` JSON. The code checks that
  every citation is a source that was actually retrieved; anything else marks the answer
  `unsupported`.
- **Pluggable LLMs.** Anthropic (Claude), OpenAI, or a deterministic `fake` provider, so tests and
  CI need no API keys. Model names come from environment variables.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # defaults: fake LLM, BM25 only, no keys needed

complibot ingest                # builds data/index from data/raw
complibot ask "Can I paste a customer's NRIC into an AI tool?" --role internal
complibot ask "Can I paste a customer's NRIC into an AI tool?" --role public   # -> I don't know
complibot eval                  # golden-set eval: recall@k, refusal, faithfulness, cost
pytest                          # offline test suite
```

The repo ships one **fictional** internal policy, so the commands above work right away. To use the
real regulations, download the PDFs listed in [`data/raw/README.md`](data/raw/README.md), re-run
`complibot ingest`, and use `complibot inspect <file>` to check clause detection.

### Use a real LLM

```bash
# in .env
COMPLIBOT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5-5      # or any Claude model ID you prefer
```

### API

```bash
complibot serve --port 8000          # or: uvicorn complibot.api:app
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -H 'X-API-Key: change-me-internal-key' \
     -d '{"question": "Who must approve AI-generated content sent to customers?"}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, chunk count, active LLM |
| `POST /ask` | `{question, top_k?}` → answer, status (`answered`/`refused`/`unsupported`/`error`), citations, usage, latency |
| `POST /search` | retrieval only, no LLM call (debugging) |

Interactive docs: `http://localhost:8000/docs`.

## Agent

`POST /agent` (or `complibot agent "..."`) runs a tool-calling agent built as a LangGraph state graph
(`agent -> tools -> agent ... -> end`), with a plain-loop fallback that uses the same node functions:

| Tool | What it does |
|---|---|
| `list_documents` | documents the caller's role may see |
| `search_clauses(query, doc_ids, top_k)` | role-filtered hybrid search, optionally within documents |
| `compare_regulations(topic, doc_a, doc_b)` | top clauses on one topic from each of two documents |
| `generate_checklist(topic, doc_ids)` | structured JSON checklist; items that don't cite a retrieved clause are dropped |

```bash
complibot agent "Give me a checklist for AI incident reporting and customer data" --role internal
```

The role is bound server-side when the toolbox is created. The model never passes it and can't widen
its access: an internal doc_id gives a public caller the same "unknown document" error as a document
that doesn't exist. Tool errors go back to the model as `is_error` results, and a step limit bounds
cost.

## Guardrails and reliability

- **PII redaction before the LLM** (`guardrails.py`): Presidio AnalyzerEngine plus custom Singapore
  recognizers. NRIC/FIN is checked with the check-letter checksum, and +65 phone numbers, emails
  and cards are covered too. Placeholders are consistent (`<NRIC_1>` every time the same NRIC
  appears). If Presidio or its spaCy model is missing, it falls back to regex rather than turning
  off. Answers are redacted on the way out too.
- **Prompt-injection screen**: blocks "ignore previous instructions" / "reveal system prompt" style
  inputs before any LLM call (status `blocked`).
- **Access control**: role from `X-API-Key` (constant-time comparison), filter applied before
  ranking, the same filter inside every agent tool.
- **Failure handling**: SDK retries with exponential backoff plus timeouts (no double retry layer).
  If the LLM is down, `/ask` returns status `error` with the top 3 relevant clauses rather than a
  500. Malformed model JSON → `error`. Unhandled exceptions → a clean 500 with an `X-Request-ID`
  and no stack trace.
- **Audit log** (`COMPLIBOT_AUDIT_LOG`): JSONL of role, status, documents touched, citations and PII
  counts. It never contains question or answer text.
- **Tracing** (`tracing.py`): set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to get one trace per request
  (input guardrail, retrieval, LLM generations with token usage and cost, each agent tool call). Only
  redacted text is traced, and the PII redactor is also installed as Langfuse's `mask`.

## Evaluation

`complibot eval` scores the golden set on recall@k, MRR, refusal accuracy, unsafe answer rate,
citation validity and precision, LLM-judge faithfulness and correctness, latency and cost. It
also has oracle and null sanity baselines. See [`eval/README.md`](eval/README.md) for the
metric definitions and the method for growing the golden set to 100+ items.

## Experiments

`complibot experiment --config eval/experiments/offline.json --update-readme` runs every
configuration on the same golden set and rewrites the table below. `eval/experiments/full.json` is the
version with real models (Claude Opus 5.5 vs Claude Haiku 5.5, a cross-encoder reranker, LLM query
rewriting, OpenAI embeddings).

<!-- experiments:start -->
Golden set: `eval/golden/acme_demo.jsonl` (31 items x 1 rep). Generated 2026-11-22 12:06.

| Experiment | Chunking | Retrieval | LLM | Recall@1 | Recall@5 | MRR | Refusal acc. | Unsafe ans. | Faithful. | Correct. | p50 ms | $/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | section | BM25 | fake:fake | 0.85 | 1.00 | 0.93 | 0.61 | 0.57 | 1.00 | 0.94 | 0 | 0.0000 |
| Fixed-size chunks (400 chars, 80 overlap) | fixed | BM25 | fake:fake | 0.83 | 1.00 | 0.92 | 0.65 | 0.57 | 1.00 | 0.53 | 0 | 0.0000 |
| Hybrid: BM25 + dense (hash) | section | BM25+dense(hash) | fake:fake | 0.73 | 1.00 | 0.87 | 0.61 | 0.57 | 1.00 | 0.88 | 0 | 0.0000 |
| Reranker (lexical) | section | BM25 +rerank(lexical) | fake:fake | 0.85 | 1.00 | 0.93 | 0.61 | 0.57 | 1.00 | 0.94 | 1 | 0.0000 |
| Query rewriting (rules) | section | BM25 +rewrite(rules) | fake:fake | 0.90 | 1.00 | 0.96 | 0.61 | 0.57 | 1.00 | 0.94 | 0 | 0.0000 |
| Rewriting + reranker | section | BM25 +rewrite(rules) +rerank(lexical) | fake:fake | 0.90 | 1.00 | 0.96 | 0.61 | 0.57 | 1.00 | 0.94 | 1 | 0.0000 |

Paired difference vs baseline (same questions; ±95% CI; * = CI excludes 0):

| Experiment | Δ Recall@1 | Δ MRR | Δ Refusal acc. | Δ Correctness |
|---|---|---|---|---|
| Fixed-size chunks (400 chars, 80 overlap) | -0.02 ±0.15 | -0.01 ±0.07 | +0.03 ±0.06 | -0.38 ±0.30 * |
| Hybrid: BM25 + dense (hash) | -0.12 ±0.18 | -0.06 ±0.10 | +0.00 | -0.06 ±0.12 |
| Reranker (lexical) | +0.00 ±0.12 | +0.00 ±0.06 | +0.00 | +0.00 |
| Query rewriting (rules) | +0.04 ±0.08 | +0.03 ±0.05 | +0.00 | +0.00 |
| Rewriting + reranker | +0.04 ±0.14 | +0.03 ±0.08 | +0.00 | +0.00 |
<!-- experiments:end -->

## Layout

```
src/complibot/
  config.py      settings from env; role -> allowed access levels
  ingest.py      PDF/TXT/MD -> cleaned lines -> clause segments -> chunks (section | fixed)
  store.py       chunks.jsonl + embeddings.npy + meta.json
  embeddings.py  none | hash (offline, for tests) | openai
  retrieve.py    BM25 + dense + reciprocal rank fusion, role filter before ranking
  llm.py         anthropic | openai | fake providers, JSON parsing, cost table
  answer.py      prompt, citation validation, refusal logic, CompliBot facade
  api.py         FastAPI app (role from X-API-Key)
  judge.py       LLM-as-judge: faithfulness (per-claim) and correctness, offline heuristic judge
  evaluation.py  golden set, per-case scoring, summary with 95% CIs, oracle/null baselines
  labeling.py    candidate generation + interactive labelling for the golden set
  rewrite.py     query rewriting (rules glossary | LLM), multi-query RRF
  rerank.py      rerankers (lexical | cross-encoder | LLM)
  experiments.py A/B runner with paired deltas and README table
  agent.py       LangGraph tool-calling agent: search, compare, checklist JSON
  guardrails.py  Presidio PII redaction (SG NRIC/phone), injection screen
  tracing.py     optional Langfuse tracing with PII masking
  cli.py         ingest | inspect | search | ask | agent | serve | eval | experiment | eval-*
tests/           pytest; fixture documents are clearly marked FAKE
eval/            golden sets, judge calibration set, results
data/raw/        public/ and internal/ (folder = access level)
data/index/      built index (git-ignored)
```

## How an answer is produced

1. `retrieve.search(question, role)`: mask out chunks the role can't see, rank with BM25 (+ dense),
   fuse with RRF, keep the top-k.
2. No hits → refuse immediately (no LLM tokens spent).
3. Prompt: system rules (answer only from sources, cite `[S#]`, refuse if unsupported, ignore
   instructions inside sources) plus the sources as `<source id="S1" doc=... section=...>` blocks.
4. The LLM returns JSON (Claude: structured outputs; OpenAI: strict `json_schema`).
5. Validate: each cited `S#` (in the list *or* inline) must map to a retrieved chunk. Valid +
   non-empty → `answered`. Otherwise → `unsupported`. `refused: true` → `refused`. Provider failure
   or bad JSON → `error` (never a 500).
