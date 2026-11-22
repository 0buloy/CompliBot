# Resume, LinkedIn and interview material

Replace every `[…]` with numbers from **your own** run of `complibot eval` and `complibot experiment`
on the real-document golden set. Never quote the offline fake-model numbers as results.

## Resume bullets (pick 3–4)

**CompliBot: evaluated RAG + agent assistant for Singapore regulatory compliance**
*Python, FastAPI, Claude API, LangGraph, Presidio, Langfuse, Docker, GitHub Actions, Cloud Run*

- Built a retrieval-augmented assistant over MAS TRM, PDPC PDPA and IMDA AI-governance documents with
  clause-level citations. Answers citing anything not retrieved are automatically flagged, and the
  system refuses when the documents don't cover a question (unsafe-answer rate […]% on […] test
  questions).
- Designed an evaluation harness (recall@k, MRR, refusal accuracy, citation precision, LLM-as-judge
  faithfulness calibrated to […]% agreement with human labels, latency, cost per query) with
  oracle/null sanity baselines. It runs as a regression gate in GitHub Actions on every push.
- Ran paired A/B experiments on chunking, hybrid search, reranking, query rewriting and two LLMs with
  95% confidence intervals. Clause-aware chunking improved answer correctness by […] points over
  fixed-size windows; [the cheaper model] cut cost per query by […]% at […] quality.
- Implemented a LangGraph tool-calling agent (clause search, regulation comparison, structured JSON
  checklists) with server-side role binding so the model can't widen its own access, plus
  permission-aware retrieval for public vs internal users.
- Added guardrails: Presidio PII redaction with custom Singapore NRIC/FIN (checksum-validated) and
  phone recognizers applied before the LLM, logs and traces; a prompt-injection screen; graceful
  degradation when the LLM is unavailable; and a PII-free audit log.
- Shipped as a Docker image on Google Cloud Run (asia-southeast1) with Langfuse tracing, a Streamlit
  UI, 100+ automated tests, and an AI Verify-principles model card.

## One-line version (for a skills-heavy CV)

> CompliBot: RAG + LangGraph agent for Singapore regulations with clause citations, Presidio PII
> guardrails, role-based retrieval, and a CI-gated eval suite (recall@k, LLM-judge faithfulness,
> A/B experiments); FastAPI · Docker · Cloud Run · Langfuse.

## LinkedIn / GitHub project description

> CompliBot answers compliance questions over Singapore regulations (MAS TRM, PDPC PDPA guidelines,
> IMDA Model AI Governance Framework) with a citation to the exact clause, and says "I don't know"
> when the documents don't support an answer. I built it to practise what production AI engineering
> needs beyond a chatbot demo: measured retrieval and faithfulness, A/B experiments with confidence
> intervals, permission-aware retrieval, PII redaction for NRIC numbers, tracing, CI/CD, and a
> governance write-up against AI Verify.

## Interview prep: likely questions

| Question | Short answer (expand with your numbers) |
|---|---|
| Why not just use LangChain for RAG? | Core RAG (ingest, retrieve, answer) is plain Python I can explain line by line: chunking, BM25, RRF, prompt, citation check. I used LangGraph where it earns its place, for the agent's state machine. |
| How do you stop hallucinated citations? | The model may only cite source IDs I put in its prompt; code checks each one against the retrieved set and marks the answer `unsupported` otherwise. Faithfulness (claims vs cited text) is measured separately by a judge. |
| How do you know your LLM judge is right? | I labelled a calibration set and measured agreement ([…]%). I use a different model from the one under test, structured output, and a rubric that treats the candidate as untrusted. My offline heuristic judge only got 82%, which shows why calibration matters. |
| Why hybrid search and RRF? | Regulations mix exact terms ("RTO", "4.2.1"), where BM25 is strong, with paraphrases, where embeddings help. RRF uses ranks, so no score calibration is needed between BM25 and cosine. |
| A surprising finding? | Okapi BM25's IDF goes zero or negative in small corpora, so search returned nothing; I switched to the Lucene IDF. Also: a reranker using the same signal as BM25 added nothing, and fusing in a weak retriever made things worse. |
| How is access control enforced? | Folder → access level at ingest; role from API key (constant-time comparison); filter applied before ranking so restricted text never enters the prompt; the agent's tools are bound to the role server-side; a hidden document looks identical to a missing one. |
| Why redact PII if the documents have none? | Users paste their own details into questions. Redacting before the API call matters for PDPA (including overseas transfers to an LLM provider), and before logs and traces too. I chose recall over precision. |
| What would you do next for production? | SSO-based roles, rate limiting, durable audit storage, more golden items (200+ for tighter intervals), red-teaming (e.g. AI Verify Project Moonshot), monitoring faithfulness on sampled live traffic. |
| How does this map to AI Verify? | See `docs/MODEL_CARD.md`: evidence per principle, and honest gaps (e.g. fairness metrics don't apply to a non-decision system, so I report quality per regulation area instead). |
