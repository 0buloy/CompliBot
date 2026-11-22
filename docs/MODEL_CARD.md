# CompliBot: system card (AI Verify-style self-assessment)

> **What this is:** a self-assessment of CompliBot against the 11 AI governance principles used by
> Singapore's **AI Verify** testing framework (IMDA / AI Verify Foundation), plus the IMDA **Model AI
> Governance Framework**. It is **not** an AI Verify certification or an official report. Each claim
> points to code, a test or an eval metric so a reviewer can check it, and gaps are listed openly.

| | |
|---|---|
| System | CompliBot v0.1: retrieval-augmented assistant and tool-calling agent |
| Owner | _your name_ |
| Underlying models | Configurable. Default `claude-opus-5-5` (Anthropic) for answers and `claude-sonnet-5-5` as the eval judge; OpenAI supported. No model is trained or fine-tuned by this project. |
| Date / version | Fill in at release (git SHA is recorded in every eval run's `summary.json`) |

## 1. Intended use

- **Users:** compliance, risk and technology staff in Singapore financial and technology
  organisations.
- **Task:** find and summarise what specific clauses of MAS, PDPC and IMDA documents (and internal
  policies) say, with clause-level citations. Produce draft compliance checklists and side-by-side
  comparisons.
- **Decision role:** decision support only. A human compliance officer reads the cited clauses and
  makes the decision (human-in-the-loop, in the Model AI Governance Framework's terms).

**Out of scope:** legal advice; questions about documents not in the corpus; automated compliance
decisions; any use where the answer is acted on without reading the cited clause.

## 2. Data

| Source | Access level | Notes |
|---|---|---|
| MAS Technology Risk Management Guidelines (2021) | internal (simulated) | public document; `internal` shows the access-control design |
| PDPC Advisory Guidelines: Key Concepts in the PDPA; NRIC; AI Recommendation & Decision Systems | public / internal | |
| IMDA/PDPC Model AI Governance Framework (2nd ed.); Model AI Gov. Framework for Generative AI (2024) | public | |
| `acme-ai-usage-policy.md` | internal | **fictional**, written for this project |

- No personal data is in the corpus. User questions may contain personal data, which is redacted
  before any LLM call, log or trace (see Security).
- Documents are processed at ingest: page text is extracted, then split into clause-level chunks
  with document, clause, heading, page and access-level metadata.

## 3. How it works

Question → PII redaction and injection screen → role-filtered hybrid retrieval (BM25 + optional dense,
RRF, optional reranker and query rewriting) → LLM answers in JSON citing source IDs → citation check
(every cited ID must be a retrieved clause) → output redaction → audit log and trace. The agent adds
tools (search, compare, checklist) under the same role binding and citation check.

## 4. Evaluation summary

Full method in [`eval/README.md`](../eval/README.md); experiment results in the main README.

| Metric | What it guards | Where |
|---|---|---|
| recall@k, MRR | the right clause is retrieved | `complibot eval` |
| refusal accuracy, unsafe-answer rate | says "I don't know" when the corpus doesn't answer | `complibot eval` |
| supported rate, citation precision | citations are real and point to the right clause | `complibot eval` |
| faithfulness (LLM judge, per claim) | the answer says only what the cited text says | `complibot eval --judge-provider anthropic` |
| judge–human agreement | the judge itself is trustworthy | `complibot eval-calibrate` |
| harness sanity | oracle scores 1.0, null fails | CI `eval` job |

> Fill in after running on your labelled real-document golden set:
> recall@5 = __, unsafe-answer rate = __, faithfulness = __, judge agreement = __ (n = __ items, __ reps).

## 5. AI Verify principles: self-assessment

| # | Principle | What CompliBot does | Evidence | Gaps / next steps |
|---|---|---|---|---|
| 1 | **Transparency** | Every answer shows its sources (document, clause, page). The UI says "Not legal advice". This card and the eval method are published. | `answer.py` citations; `ui/app.py` | Add a user-facing notice of which LLM provider processes questions |
| 2 | **Explainability** | Answers are traceable to specific clauses. The agent's tool calls are returned as `steps`. A status says *why* there was no answer (refused / unsupported / blocked / error). | `AgentResult.steps`; status field | No per-sentence attribution yet |
| 3 | **Repeatability / reproducibility** | Deterministic retrieval; index built from versioned files; eval runs record config, git SHA, model and per-case rows; offline eval is deterministic in CI. | `eval/results/*/summary.json`; CI | LLM sampling isn't deterministic: report ±CI over reps |
| 4 | **Safety** | Refuses when the documents don't support an answer; unsupported answers are flagged; refusal is measured. | `IDK` path; `unsafe_answer_rate` | Expand the near-miss and out-of-scope golden items |
| 5 | **Security** | PII redaction (NRIC/FIN with checksum, +65 phones, email, cards) before LLM, logs and traces; prompt-injection screen; sources treated as data in prompts; secrets only from env / Secret Manager; non-root container; clean 500s. | `guardrails.py`; `tests/test_guardrails.py`; `Dockerfile` | Pattern-based injection screening is a first layer only; red-team with a tool such as AI Verify Project Moonshot |
| 6 | **Robustness** | Retries with backoff and timeouts; degraded mode returns relevant clauses when the LLM is down; malformed model output handled; agent step limit; tool errors returned to the model. | `llm.py`; `answer.py`; `tests/test_agent.py` | Load testing; paraphrase-robustness set is small |
| 7 | **Fairness** | Not a decision system about individuals, so classic group-fairness metrics don't apply. Answer quality is reported per document and category so no regulation area is silently worse. | `summary.by_category` | Check quality on questions phrased in Singlish or non-native English |
| 8 | **Data governance** | Document provenance in the catalog; access level from folder (unknown folder → internal, fail closed); no PII in the corpus; audit log stores counts and IDs, never text. | `ingest.access_level_for`; `write_audit` | Retention policy for audit logs; document version tracking |
| 9 | **Accountability** | Named owner (this card); audit log of who asked (role), what was touched and what was returned; CI gate blocks regressions. | `COMPLIBOT_AUDIT_LOG`; `.github/workflows/ci.yml` | Incident-response runbook |
| 10 | **Human agency & oversight** | Human-in-the-loop by design: the system drafts, cites and refuses; a person decides. Checklists are drafts that cite their sources. | Intended use; UI | Feedback button to flag bad answers into the golden set |
| 11 | **Inclusive growth, societal & environmental well-being** | Makes dense public regulations easier to find and check; runs on scale-to-zero infrastructure; offline/BM25 mode needs no GPU. | `docs/DEPLOY.md` | Measure token and energy use per answer |

## 6. Model AI Governance Framework (IMDA/PDPC, 2nd ed.): alignment

| Area | CompliBot |
|---|---|
| Internal governance structures and measures | Owner on this card; CI eval gate; changes to models/prompts require a re-run of the golden set |
| Determining the level of human involvement | Human-in-the-loop: wrong answers have high severity, but a human reviews the cited clause before acting |
| Operations management | Data lineage (doc → clause → chunk), versioned golden set, eval and experiment tracking, tracing |
| Stakeholder interaction and communication | Citations, status explanations, "not legal advice" notice, this card |

The **Model AI Governance Framework for Generative AI (2024)** dimensions most relevant here:
*accountability* (above), *data* (provenance, PII handling), *trusted development and deployment*
(eval before release, this card), *incident reporting* (audit log + tracing), *testing and assurance*
(golden set, judge calibration, CI gate) and *security* (guardrails).

## 7. Known limitations

- Clause detection is heuristic. Unusual layouts (tables, multi-column, scanned PDFs) can merge or
  split clauses. Check each new document with `complibot inspect`.
- The LLM can still paraphrase a clause slightly wrongly even when the citation is valid. That's
  why faithfulness is measured and the human reads the clause.
- The offline "fake" LLM and heuristic judge exist for CI only; their scores say nothing about real
  answer quality.
- Documents go out of date. Answers reflect the corpus version, not the current law.
