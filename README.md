# CompliBot

An **evaluated** RAG assistant that answers compliance questions over Singapore regulatory
documents (MAS Technology Risk Management Guidelines, PDPC PDPA advisory guidelines, IMDA Model AI
Governance Framework), with clause-level citations on every answer and an "I don't know" when the
documents don't support one.

> Status: Phase 1 (core RAG) is done. Evaluation, experiments, agent + guardrails, and deployment are
> coming in later phases.

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
pytest                          # 52 tests, offline
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
  cli.py         ingest | inspect | search | ask | serve
tests/           pytest; fixture documents are clearly marked FAKE
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
