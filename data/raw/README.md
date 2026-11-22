# Source documents

**The folder decides who can see a document:**

| Folder | Access level | Visible to |
|---|---|---|
| `data/raw/public/` | `public` | everyone |
| `data/raw/internal/` | `internal` | callers with an internal API key (and the CLI with `--role internal`) |

All of these are public documents. The `internal` split simulates a firm restricting some
material (e.g. supervisory guidance used by its compliance team, plus its own internal policy) to
staff. `internal/acme-ai-usage-policy.md` is a **fictional** policy written for this project.

PDFs are git-ignored: download them yourself and **save them under exactly these file names**
(the names map to titles in `catalog.json`).

| Save as | Document | Where to get it |
|---|---|---|
| `public/pdpc-ag-key-concepts-pdpa.pdf` | PDPC *Advisory Guidelines on Key Concepts in the Personal Data Protection Act* (latest revision) | pdpc.gov.sg → Guidelines & Consultation → Advisory Guidelines |
| `public/pdpc-ag-nric.pdf` | PDPC *Advisory Guidelines for NRIC and Other National Identification Numbers* | pdpc.gov.sg → Advisory Guidelines (sector/topic-specific) |
| `public/imda-model-ai-gov-framework-2e.pdf` | IMDA/PDPC *Model AI Governance Framework, Second Edition* (Jan 2020) | pdpc.gov.sg → Help & Resources → Model AI Governance Framework |
| `public/imda-genai-gov-framework-2024.pdf` | *Model AI Governance Framework for Generative AI* (30 May 2024) | aiverifyfoundation.sg → Resources |
| `internal/mas-trm-guidelines-2021.pdf` | MAS *Technology Risk Management Guidelines* (18 January 2021) | mas.gov.sg → Regulation → Guidelines → "Technology Risk Management Guidelines" |
| `internal/pdpc-ag-ai-recommendation-systems.pdf` | PDPC *Advisory Guidelines on Use of Personal Data in AI Recommendation and Decision Systems* (1 March 2024) | pdpc.gov.sg → Advisory Guidelines |

Tip: search the exact title on the agency's site if a link has moved; agencies re-organise their
pages often. After downloading, check clause detection on each file:

```bash
complibot inspect data/raw/internal/mas-trm-guidelines-2021.pdf --limit 40
```

You should see clause IDs such as `3.1.1`, `3.1.2` with the right heading titles and pages. If a
document uses a layout the heuristics miss, tune `classify_line()` in `src/complibot/ingest.py` and
add a test for it in `tests/test_ingest.py`.
