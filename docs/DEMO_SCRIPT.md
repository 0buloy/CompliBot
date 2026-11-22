# Demo video script (about 3 minutes)

**Setup before recording:** the real PDFs are indexed, `COMPLIBOT_LLM_PROVIDER=anthropic`, the API is
running, the Streamlit UI is open (local mode with the role toggle), and a terminal is ready. Record
at 1080p and zoom the browser to 125%.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | README title + architecture diagram | "This is CompliBot, an assistant for Singapore compliance questions over MAS, PDPC and IMDA documents. Every answer cites the exact clause, and when the documents don't cover something it says so. I'll show the answers, the security, and how I measure it." |
| 0:15–0:45 | UI, role **public**: *"When must an organisation notify the PDPC of a data breach?"* | "The answer cites the clause and page. Expand a source to see the original text. The model only gets the retrieved clauses, and my code checks every citation against what was retrieved. A made-up citation gets flagged as unsupported." |
| 0:45–1:05 | Ask *"What are the EU AI Act's prohibited practices?"* | "That's out of scope, so it refuses instead of answering from general knowledge. In compliance a confident wrong answer is the costly failure, so I measure the unsafe-answer rate separately." |
| 1:05–1:30 | Role **public**: *"What recovery time objective does MAS expect for critical systems?"* → refused. Switch to **internal** → answered. | "Access control happens before retrieval. The MAS document is internal in this setup, so a public user's search never even scores those clauses. Through the API the role comes from the API key, not from anything the client sends." |
| 1:30–1:50 | *"My NRIC is S1234567D, can my bank keep a copy of my IC?"* | "The NRIC is redacted by Presidio with a custom Singapore recognizer that validates the NRIC checksum, before anything reaches the LLM, the logs or the traces. You can see what the model actually saw." |
| 1:50–2:20 | Mode **Agent**: *"Make a checklist for third-party technology service providers"* | "In agent mode the model calls tools: search, compare two regulations, and generate a checklist. The checklist comes back as structured JSON, and any item that doesn't cite a retrieved clause gets dropped. Here's the tool trace." Click download JSON. |
| 2:20–2:45 | Terminal: `complibot eval` table, then README experiments table | "This is how I know it works: a golden set with recall@k, refusal accuracy, and faithfulness from an LLM judge that I calibrated against my own labels. I A/B tested chunking, reranking, query rewriting and two models, compared question by question with confidence intervals. Section chunking beat fixed-size windows on correctness." |
| 2:45–3:00 | GitHub Actions run, `docs/MODEL_CARD.md` | "Every push runs the tests and the eval as a regression gate, and the Docker image is built and smoke-tested. It's deployed on Cloud Run in the Singapore region, and I wrote a model card against the AI Verify principles. Thanks for watching." |

**Tips:** pre-warm the API with one request so there's no cold start on camera. Keep the questions in
a text file to paste. Cut long LLM waits in editing. Show real numbers from your own eval run, never
the offline fake-model numbers.
