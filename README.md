# Claims Intake Copilot

An AI copilot that helps insurance claims adjusters triage claim packets: it sorts every page by document type and answers questions about the claim with page-level citations.

## Problem

A claim arrives as a **packet**: claim forms, damage photos, police reports, repair estimates, medical bills and letters, often scanned and in no order. Before an adjuster can make any decision, they spend about **50 minutes per claim** sorting pages and searching for key facts (policy number, date of loss, fault, amounts). That slows claims, costs adjuster time on admin work, and missed pages cause wrong decisions and rework.

## Users

- **Primary:** insurance claims adjusters (property and auto).
- **Secondary:** claims team leads who track triage time and quality.

## What it does

1. **Ingest:** takes a claim packet (PDF / scanned images) and extracts text per page with OCR, keeping page numbers.
2. **Classify:** a fine-tuned PyTorch page classifier routes each page to a type: claim form, photo, police report, repair estimate, medical bill, letter/email, other. Low-confidence pages are flagged for a human.
3. **Index:** chunks and embeds page text into a vector database, filtered by `claim_id` so a search only sees that claim's pages.
4. **Answer:** a LangChain RAG app answers adjuster questions ("What is the repair estimate?") using only retrieved pages, with the page number cited. If the answer is not in the packet, it says so instead of guessing.

## Success metrics

| Metric | What it measures | Target |
|---|---|---|
| Page-routing accuracy | Classifier labels pages correctly | ≥ 90% |
| Retrieval recall@5 | Correct page is in the top 5 results | ≥ 0.90 |
| Answer faithfulness (RAGAS) | Answers are grounded in the cited pages | ≥ 0.90 |
| Citation accuracy | Cited page actually contains the answer | ≥ 95% |
| p95 latency | Time to answer a question | < 3 s |
| Cost per claim | LLM + embedding cost per packet | tracked |
| Triage time | Adjuster minutes per claim (pilot) | 50 → ~15 |

All metrics are measured on a labeled gold set (see `eval/`).

## Out of scope

- **No claim decisions.** The copilot assists; a human adjuster always approves or denies.
- No fraud detection or payout calculation (possible future work).
- No real customer data: development uses public and synthetic documents only.

## Data

Real claims contain PII, so development uses:
- **RVL-CDIP** (public scanned-document dataset) for training the page classifier.
- **Synthetic claim packets** generated with an LLM and rendered to PDF (claim form, police report, estimate, medical bill, letters) plus public car-damage photos.
- A **gold set** of ~100 questions with correct answers and page numbers.

## Architecture (planned)

```
PDF packet -> ingest (OCR, per page) -> classify (PyTorch) -> index (chunk + embed, vector DB, claim_id filter)
                                                                         |
                                     adjuster question -> retrieve top-5 -> LLM (LangChain) -> answer + page citation
```

- **Services:** FastAPI, each component in Docker, docker-compose for local runs.
- **CI/CD:** GitHub Actions runs tests and the eval set; the build fails if accuracy drops.
- **Vector DB:** FAISS vs. Chroma vs. Pinecone compared on the same embeddings and gold set.

## Repo layout

```
ingest/    PDF -> pages -> text
classify/  PyTorch page classifier
index/     chunking, embeddings, vector DB
qa/        LangChain RAG with citations
api/       FastAPI endpoints
eval/      gold set + scoring scripts
data/      datasets (raw and synthetic, not committed)
tests/     unit tests
```

## Roadmap

- [ ] Phase 1: synthetic packet generator + gold set
- [ ] Phase 2: end-to-end baseline (upload -> classify -> index -> ask -> cited answer)
- [ ] Phase 3: measure and improve (classifier fine-tuning, vector DB comparison, hybrid search + reranking, RAGAS)
- [ ] Phase 4: production polish (Docker services, CI eval gate, latency/cost logging, PII masking, Streamlit demo)
- [ ] Phase 5: results table, architecture diagram, demo video

## Results

_To be filled in as each phase is measured._
