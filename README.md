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
- **Gold Q&A** (`eval/gold_qa.jsonl`): questions with the exact answer and two page keys: `evidence_pages` (the official source page, e.g. the claim form for the policy number) and `answer_pages` (every page where the answer is printed, e.g. the vehicle also appears on the police report and estimate). Questions about missing documents (no police report, no medical bill, no tow) have the answer `NOT_IN_PACKET`, to test that the copilot refuses instead of guessing.
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
.venv/bin/python -m pytest -q                          # 30 tests: data, page reading, sorter, search, scoring
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
  - [x] Step 2: page sorter (TF-IDF + logistic regression vs. ResNet-18), stress test, real-scan experiment (RVL-CDIP)
  - [x] Step 3: search (chunking, bge-small embeddings, Chroma vs FAISS vs BM25 vs hybrid), scored on 616 questions
  - [ ] Step 4: answer with citations (LLM over top chunks, "not in packet" refusal)
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

### Step 2: page sorter (`classify/`)

Two sorters trained on the same split and scored the same way:

| Sorter | Input | Model |
|---|---|---|
| **Text** | page text from `data/ingested/` | TF-IDF (words + 2-word phrases) + logistic regression, class-balanced |
| **Image** | page rendered as a 224x224 image | **ResNet-18** pretrained on ImageNet, last layer replaced (1000 -> 7 classes), fine-tuned 4 epochs (AdamW, weighted cross-entropy, Apple MPS GPU) |

- **Split by claim, not by page** (35 claims train / 15 test, 402 / 161 pages), so names, amounts and IDs from one claim never leak into the test set.
- **Metrics:** accuracy, macro-F1 (every class counts equally; photos are 37% of pages, medical bills 5%), confusion matrix, and the share of pages under 80% confidence that would go to a human.

```bash
.venv/bin/python -m classify.text_baseline   # trains in ~1 s, prints top words per class
.venv/bin/python -m classify.cnn             # ~25 s on an M-series Mac
.venv/bin/python -m classify.stress          # robustness test on damaged pages
```

**Clean test pages (161):** both sorters score **100% accuracy, macro-F1 1.00**, including scanned pages. Top text features are sensible: `patient`, `balance due` -> medical bill; `officer`, `driver` -> police report; `towing`, `amount paid` -> other.

That score shows the pipeline works, not that the models are robust: the synthetic templates are uniform. So the test pages were **damaged on purpose**:

| Text sorter: words kept | Accuracy | Sent to human |
|---|---|---|
| 100% | 100.0% | 0.0% |
| 50% | 97.5% | 39.8% |
| 20% | 85.7% | 97.5% |
| 10% | 73.9% | 100.0% |

| Image sorter: blur + shrink | Accuracy | Sent to human |
|---|---|---|
| none / light | 100.0% | 0.0% |
| medium | 100.0% | 25.5% |
| heavy | 65.8% | 59.6% |
| extreme | 54.7% | 65.2% |

**What this shows:** the text sorter's confidence drops with its accuracy, so nearly all its mistakes go to a human. The image sorter is **overconfident** under heavy damage: some pages are wrong *and* above 80% confidence.

### Step 2b: adding real scans (RVL-CDIP)

Synthetic pages are too uniform, so the sorters were also tested on **real scanned office documents** from **RVL-CDIP** (Harley et al., ICDAR 2015): 5 of its 16 types, 1,000 pages, mapped to our labels (letter / email / memo -> correspondence, form -> claim_form, invoice -> other). The pages were OCR'd with Tesseract (mean confidence 77.5%, vs ~88% on our simulated scans). Source and license: `data/raw/rvl_cdip/SOURCE.md` (public Legacy Tobacco Document Library, research and education use).

```bash
.venv/bin/python -m classify.rvl          # OCR the 1,000 RVL pages once (~2 min, 6 workers)
.venv/bin/python -m classify.experiment   # train A vs B, test on synthetic and real
```

- **Train A:** synthetic only (402 pages). **Train B:** synthetic + 800 real RVL pages.
- **Real test set:** 200 RVL pages from RVL's own validation split (never seen in training).
- **Combined** = average of the text and image sorters' probabilities; when they disagree, confidence drops and the page goes to a human.

| Sorter | Trained on | Synthetic test (161) | **Real test (200)** | Real: sent to human | Real: accuracy when confident |
|---|---|---|---|---|---|
| text | synthetic only | 100% | 67.5% | 100% | n/a |
| image (ResNet-18) | synthetic only | 100% | 53.0% | 65.5% | 66.7% |
| text | synthetic + real | 100% | 88.0% | 47.5% | 99.0% |
| image (ResNet-18) | synthetic + real | 100% | 84.0% | 33.5% | 90.2% |
| **combined** | synthetic + real | **100%** | **88.5%** (macro-F1 0.86) | 46.5% | **99.1%** |

**What this shows:**
1. **Domain gap.** Models trained only on synthetic pages score 100% on synthetic tests but **53-68% on real scans**. A perfect synthetic score did not mean the model works on real documents.
2. **Real data closes most of the gap**: +20.5 points (text) and +31 points (image) on real scans, with **no loss** on synthetic pages.
3. **Safe operating point:** the combined sorter is right on **99.1%** of the pages it is confident about and sends the rest (46.5%) to a human. In production, the 80% threshold would be tuned against the cost of human review.
4. **Weakest classes:** claim_form vs other (generic RVL forms vs invoices look alike). Known label noise: RVL "form" is a generic office form, not an insurance loss notice.

Next: calibrate the CNN (temperature scaling), tune the confidence threshold, and replace RVL with a small hand-labeled set of real claim pages when available.

**Page labels saved:** `python -m classify.label_pages` trains the combined sorter on synthetic training claims + real RVL scans and writes `page_type`, `page_type_conf`, `needs_review` (< 0.8) and `split` into every line of `data/ingested/<claim_id>.jsonl`. On the 15 held-out claims it labels 100% of pages correctly; search uses the label as chunk metadata and citations.

### Step 3: search (`index/`)

```bash
.venv/bin/python -m index.search --claim CLM-2026-00008 --q "What is the repair estimate total?"
.venv/bin/python -m eval.retrieval_eval      # all search setups on the 616 answerable gold questions
```

- **Chunking:** by lines (max ~400 characters, 1-line overlap), so a label and its value ("Estimate total $2,346.78") stay together. Each chunk starts with its page type (`[repair_estimate] ...`) and keeps claim_id and page for filtering and citations. 563 pages -> **989 chunks**.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384 dimensions, runs locally, no API key), query prefix as the model recommends, vectors cached.
- **Claim isolation:** every search is restricted to one claim. Chroma uses a metadata filter (`where claim_id = ...`); FAISS has no metadata filter, so it keeps one small index per claim. Tests check that no store ever returns another claim's pages.
- **Hybrid:** vector + BM25 keyword results fused with Reciprocal Rank Fusion (combines by rank, since the two score scales differ).

Results on 616 answerable questions across 50 claims:

| Search | hit@1 | recall@5 | recall@5 (strict: source page only) | MRR | ms/query |
|---|---|---|---|---|---|
| BM25 keyword | 77.1% | 100.0% | 88.5% | 0.871 | <0.1 |
| vector (FAISS) | **92.2%** | 99.7% | **96.6%** | **0.953** | 8.8 |
| vector (Chroma) | **92.2%** | 99.7% | **96.6%** | **0.953** | 9.8 |
| hybrid (Chroma + BM25, RRF) | 90.7% | **100.0%** | 93.3% | 0.951 | 10.2 |

- **recall@5 = 100%** for hybrid across all 14 question types, including answers that only exist on **OCR'd scans** (156 questions).
- **Vector search ranks best** (right page first 92% of the time). Keyword search alone ranks poorly (77%) but never misses entirely; hybrid trades a little ranking for full recall.
- **FAISS and Chroma return identical results** (same embeddings, exact search at this size); the choice between them is operational (Chroma: metadata filters and persistence; FAISS: speed at scale). Pinecone was not compared (needs a hosted account).
- **Eval bug found and fixed:** the first run showed 93% recall because the gold set listed only the official source page; the vehicle, date and policy number are also printed on the police report, estimate and letters, so correct results were marked wrong. The gold set now has both keys, and both are reported.
- **Caveat:** a claim has only ~20 chunks, so top-5 covers a quarter of a claim; hit@1 and MRR are the more telling numbers. Larger, real packets would make retrieval harder.
