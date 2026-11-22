Golden set: `eval/golden/acme_demo.jsonl` (31 items x 1 rep). Generated 2026-11-22 12:25.

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
