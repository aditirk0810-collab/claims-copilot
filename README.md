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

Real claims contain PII, so development uses synthetic packets with exact ground truth (plus **RVL-CDIP**, a public scanned-document dataset, for classifier pre-training later).

### Synthetic claim packet generator (`synth/`)

Each packet is built from one set of claim facts, so every document agrees and every answer is known:

- **Documents:** 2-page loss notice (claim form), police report (~70% of claims), 1-2 repair estimates with line items and tax, itemized medical bill with CPT/ICD-10 codes (injury claims), adjuster email and claimant letter, towing invoice (`other`), and 2-6 **real car damage photos**.
- **Real photos:** 240 real damage photos (dents, scratches) from the [car-damage-new dataset](https://huggingface.co/datasets/JijoJS/car-damage-new1) by Roboflow user jijo, **CC BY 4.0**. Download the `test` and `valid` zips into `data/raw/car_damage/photos/` (see `SOURCE.md` there). If the folder is missing, the generator falls back to drawn placeholder photos.
- **Realistic mess:** pages are shuffled, and ~30% of non-photo pages are rendered as **scans** (rasterized, tilted, blurred, noisy JPEG) with no extractable text, so the OCR step is required.
- **Labels per packet** (`labels.json`): page type, scanned flag, and ground-truth text for every page, plus gold Q&A.
- **Gold Q&A** (`eval/gold_qa.jsonl`): questions with the exact answer and the **evidence page numbers**. Questions about missing documents (no police report, no medical bill, no tow) have the answer `NOT_IN_PACKET`, to test that the copilot refuses instead of guessing.
- **Deterministic:** the same seed always produces the same packets.

Current dataset (`--n 50 --seed 42`): **50 packets, 563 pages, 679 gold questions** (616 answerable, 63 not-in-packet), 91 scanned pages.

| Page type | Pages |
|---|---|
| photo | 207 |
| claim_form | 100 |
| correspondence | 85 |
| repair_estimate | 84 |
| police_report | 34 |
| medical_bill | 27 |
| other (towing) | 26 |

Classes are imbalanced (like real packets), so the classifier is reported with per-class F1, not only accuracy.

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m synth.generate --n 50 --seed 42   # writes data/synthetic/ and eval/gold_qa.jsonl
.venv/bin/python -m pytest -q                          # 15 tests: data labels, page reading, scoring
```

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
synth/     synthetic claim packet generator + gold Q&A
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

- [x] Phase 1: synthetic packet generator + gold set
- [ ] Phase 2: end-to-end baseline (upload -> classify -> index -> ask -> cited answer)
  - [x] Step 1: page reading (text copy + Tesseract OCR), scored against ground truth
- [ ] Phase 3: measure and improve (classifier fine-tuning, vector DB comparison, hybrid search + reranking, RAGAS)
- [ ] Phase 4: production polish (Docker services, CI eval gate, latency/cost logging, PII masking, Streamlit demo)
- [ ] Phase 5: results table, architecture diagram, demo video

## Results

### Step 1: page reading (`ingest/`)

Each page takes the cheapest route that works:

| Page looks like | Detected by | Method |
|---|---|---|
| Has a text layer (20+ words) | `page.get_text("words")` | **copy text** (instant, exact) |
| One image covering 80%+ of the page, no words | image area / page area | **OCR** with Tesseract 5 (CNN + bi-LSTM + CTC), 300 dpi |
| Photo with a short caption | image covers < 80% | keep caption, don't OCR |

```bash
.venv/bin/python -m ingest.run      # writes data/ingested/<claim_id>.jsonl
.venv/bin/python -m eval.ocr_eval   # scores against labels.json
```

Results on 50 packets (563 pages), Tesseract 5.5.2:

| Method | Pages | Mean CER | Fact survival* |
|---|---|---|---|
| copy text | 265 | 0.00% | 473/473 (100%) |
| OCR (scans) | 91 | 1.20% | 143/143 (100%) |
| photo (caption only) | 207 | - | - |

- **Method choice accuracy: 100%** (every page routed to the right method).
- **Speed:** ~0.8 s per scanned page on a laptop; text pages are instant.
- *Fact survival = gold answers (amounts, IDs, dates, names) found intact in the page text. It matters more than CER: one wrong digit in an amount means a wrong answer.
- **Error analysis:** the 1.2% CER is almost all junk characters where Tesseract reads horizontal table rules as text (e.g. `ea ar ea rage`); words and numbers came through intact. Next fixes: remove ruling lines before OCR, and compare against a cloud engine (Azure Document Intelligence / AWS Textract) on harder scans.
- **Caveat:** these scans are simulated (tilt, blur, noise). Real phone photos and fax scans are harder; the plan for production is a hand-labeled sample of real pages plus confidence-based routing to a human.
