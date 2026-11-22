# Evaluation

One command runs the golden set and prints a results table:

```bash
complibot ingest
complibot eval                                   # default: eval/golden/acme_demo.jsonl
complibot eval --golden eval/golden/sg_regulations.jsonl --provider anthropic \
               --judge-provider anthropic --reps 2
```

Each run is saved to `eval/results/runs/<timestamp>-<label>/`:
`results.jsonl` (one row per case and rep, with retrieved IDs, answer, citations, judge claims),
`errors.jsonl` (infrastructure failures, kept out of the quality metrics), `summary.json`, and `summary.txt`.

## Files

| File | What |
|---|---|
| `golden/acme_demo.jsonl` | 24 **verified** items over the fictional in-repo policy. Runs offline in CI. |
| `golden/sg_regulations.draft.jsonl` | 26 **draft** questions for the real MAS/PDPC/IMDA PDFs. Clause IDs and answers are added by `eval-label` against your downloaded PDFs. |
| `golden/judge_calibration.jsonl` | Human faithfulness labels for checking the judge (`complibot eval-calibrate`). |

## Golden item format

```json
{"id": "acme-007",
 "question": "Who has to check AI-generated messages before they go out to customers?",
 "role": "internal",
 "answerable": true,
 "source_clauses": [{"doc_id": "acme-ai-usage-policy", "section_id": "4.1"}],
 "reference_answer": "A human employee must review and approve the content before it is sent.",
 "tags": ["human-oversight"],
 "verified": true}
```

- `answerable` means answerable **for that role**. An internal-only fact asked as `public` is
  `answerable: false` (tag it `role-gated`).
- `source_clauses` lists every clause needed for a full answer (multi-clause questions list several).
- Only `verified: true` items are scored. Drafts and candidates never count until a human labels them.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| recall@k (k=1,3,5,10) | share of gold clauses in the top-k retrieved | retrieval sets the ceiling: the LLM can't cite what wasn't retrieved |
| MRR | 1 / rank of the first gold clause | rewards ranking the right clause first |
| refusal accuracy | refused exactly when `answerable` is false | the "I don't know" behaviour |
| false refusal rate | answerable items that were refused | over-cautious |
| unsafe answer rate | unanswerable items that got an answer | hallucination risk, the costly failure in compliance |
| supported rate | answers whose citations are all real retrieved chunks | the code-level citation check |
| citation precision | cited chunks that cover a gold clause | cites the *right* clause, not just *a* real one |
| faithfulness | LLM judge: claims supported by the cited text / all claims | catches answers that cite real clauses but say something the clause doesn't |
| correctness | LLM judge vs reference: yes=1, partial=0.5, no=0 | does it actually answer the question? |
| latency p50/p95 | end-to-end and LLM-only | ops |
| cost | from API token usage × price table; judge cost reported separately | so judge spend doesn't hide differences between systems |

Means include a ±95% confidence interval. Treat a difference smaller than the interval as noise.
Rough noise floor for a pass rate: `1/sqrt(n·reps)`, so 100 items × 2 reps ≈ ±7 points. This is the
main reason to grow the set to 100+ items.

### Sanity baselines

```bash
complibot eval --system oracle --no-judge   # should be ~1.0 on everything
complibot eval --system null                # should fail every answerable item
```

If the oracle isn't ~1.0, the harness or the labels are broken. If null doesn't fail, the grader is
too lenient. Run both after any change to the eval code.

### The judge

- Faithfulness and correctness are **separate calls**, each with a concrete rubric and structured
  JSON output. The candidate answer is marked as untrusted data, and the judge is told not to
  reward length.
- Use a **different model from the one under test** (`COMPLIBOT_JUDGE_MODEL`, default
  `claude-sonnet-5-5`) to limit self-preference.
- `COMPLIBOT_JUDGE_PROVIDER=fake` is a token-overlap heuristic for CI only. On the calibration set it
  agrees with humans on just 82% of cases: it misses contradictions that reuse the source's words.
  That gap is the argument for a real judge.
- **Calibrate before trusting it:** `complibot eval-calibrate --judge-provider anthropic`. Aim for
  ≥ 90% agreement, and add your own labelled examples to `judge_calibration.jsonl` (about 30–50).

## Writing the rest of the golden set (target: 100–150 items)

Aim for this mix: about 60% single-clause answerable, 15% multi-clause/comparison, 15%
unanswerable (out of scope or near miss), and 10% role-gated. Spread items across all six documents.

1. **Download the PDFs and index them.** See `data/raw/README.md`, then run `complibot ingest`. Check
   chunking with `complibot inspect <pdf>`.
2. **Label the 26 drafts.**
   ```bash
   complibot eval-label eval/golden/sg_regulations.draft.jsonl --out eval/golden/sg_regulations.jsonl
   ```
   For each question you see the top 8 clauses for that role. Type the numbers of the clauses that
   answer it (e.g. `2,5`), or `u` if nothing does. Then write a 1–2 sentence reference answer **from
   the clause text**. `e` edits the question, `r` switches role, `s` skips, `q` saves and quits.
3. **Generate candidates from the clauses** (the source is known by construction):
   ```bash
   COMPLIBOT_LLM_PROVIDER=anthropic complibot eval-generate --per-doc 15 --out eval/golden/candidates.jsonl
   complibot eval-label eval/golden/candidates.jsonl --out eval/golden/sg_regulations.jsonl
   ```
   Each sampled clause yields one answerable question and one **near-miss** (same topic, not
   answered by that clause). When labelling a near miss, check that *no other* clause answers it
   before you press `u`. Reject vague or leading questions: delete them or `e` to rewrite.
4. **Hand-write about 25 hard items** the generator won't produce:
   - comparisons: "How do MAS TRM and the Model AI Governance Framework each treat human oversight?"
   - different wording from the document: "Do I need a backup site?" rather than "alternate site"
   - role-gated: internal-document questions asked as `public`
   - out of scope: EU AI Act, other countries' regulators, legal penalties not in the documents
5. **Freeze and version.** Commit the golden set. Change it only in a deliberate commit, because
   scores before and after a golden-set change are not comparable.

Budget: about 1.5 minutes per item with the label tool, so roughly 3 hours for 120 items. It's the
highest-value hour-for-hour work in the project.
