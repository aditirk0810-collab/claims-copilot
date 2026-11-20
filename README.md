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
4. **Answer:** a LangChain RAG chain answers adjuster questions ("What is the repair estimate?") using only retrieved pages, with the page number cited. If the answer is not in the packet, it says so instead of guessing.

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
- **Planted inconsistencies** (26 of 50 claims, one each): police report date differs from the loss notice, loss notice says no injuries but a medical bill is present, a repair estimate lists a different vehicle, or medical treatment is dated before the accident. Injected with a separate random stream, so the other claims are unchanged.
- **Deep gold questions** (`eval/gold_deep_qa.jsonl`, 405): each needs 2+ documents and often math or a comparison: total cost, estimate gap, days to treatment, days to acknowledgement, was our policyholder at fault, date / injury / vehicle consistency, treatment before the accident, and a full red-flag review.
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
.venv/bin/python -m pytest -q                          # 52 tests: data, page reading, sorter, search, answers, agent, LangChain, scoring
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
  - [x] Step 4: answer with citations (Gemini over top chunks, code-side verification, "not in packet" refusal)
  - [x] Step 5: multi-document agent for deep questions (tools, code-side math, verified citations)
  - [x] LangChain (retriever + LCEL chain) and LangGraph (create_agent) implementation; demo web app
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

### Step 4: answers with citations (`qa/`)

```bash
cp .env.example .env    # then paste a free Gemini key from aistudio.google.com
.venv/bin/python -m qa.answer --claim CLM-2026-00008 --q "What is the repair estimate total?"
.venv/bin/python -m eval.qa_eval              # 15 held-out test claims
```

1. **Retrieve:** hybrid search, top 5 chunks from this claim only.
2. **Generate:** Gemini (temperature 0, JSON schema output) answers only from the numbered sources, returning `answer`, an exact `quote`, `source_pages`, or `not_in_packet: true`.
3. **Verify in code, not in the model:** the cited pages must be among the retrieved pages, the quote must appear on a cited page, and the answer must appear on the cited page (or inside that verified quote). Anything that fails is shown as **needs review**, never as fact.
4. **Cost control:** every call is cached on disk (re-running the eval is free), calls are rate-limited for the free tier, and 429 errors are retried with backoff.

Unit tests use a fake LLM to prove the checks: a made-up amount, a wrong-page citation, or a citation to a page that was not retrieved all go to review. One test caught a real bug: the first version trusted the model's own quote, so a made-up quote could "prove" a made-up answer.

Model: Gemini 3.5 Flash-Lite (Gemini 3.8 Flash answered correctly but took 76-149 s per call; Flash-Lite takes under 1 s).

### Step 5: deep questions with a multi-document agent (`qa/agent.py`)

Questions like "total cost = cheaper estimate + medical + towing" or "do the loss notice and police report agree on the date?" need several documents and arithmetic. A single search + single answer cannot do that, so an agent works in steps (max 10):

| Tool | What it does |
|---|---|
| `search(query, page_type?)` | hybrid search in this claim, optionally one document type (uses the Step 2 page labels) |
| `read_page(page)` | full page text (whole estimates, tables) |
| `calculate(expression)` | arithmetic in code (AST-parsed: numbers and + - * / only, no `eval`) |
| `days_between(start, end)` | date math in code |
| `final_answer(answer, explanation, citations)` | one exact quote per fact used |

Gemini function calling with mode `ANY` (must call a tool every turn), temperature 0, a table of contents of page types in the first message. **Code verifies the result:** every cited page must have been opened by the agent, every quote must be on its page (excerpts joined by `...` are checked one by one), and every number in the answer must be printed on a cited page or produced by `calculate` / `days_between`. Otherwise: needs review.

```bash
.venv/bin/python -m qa.agent --claim CLM-2026-00005 --q "Do the loss notice and the police report agree on the date of the accident?"
.venv/bin/python -m eval.deep_eval     # agent vs. simple copilot, 4 questions per type, balanced answers
```

Results on 40 deep questions (10 types, 4 each, balanced so planted inconsistencies are included), Gemini 3.5 Flash-Lite:

| System | Correct + verified | Correct (any status) | Needs review | Avg steps | Avg documents cited | Avg latency |
|---|---|---|---|---|---|---|
| **Agent** | **77.5%** | **92.5%** | 17.5% | 5.1 | 2.4 | 8.1 s |
| Simple (1 search + 1 answer) | 25.0% | 52.5% | 35.0% | 1 | 1.1 | ~1 s |

| Question type | Agent | Simple |
|---|---|---|
| total cost (3-4 documents + addition) | 100% | 25% |
| gap between two estimates | 100% | 0% |
| days accident -> treatment | 100% | 25% |
| days loss -> acknowledgement | 100% | 75% |
| dates consistent (loss notice vs police report) | 100% | 25% |
| same vehicle on every estimate | 100% | 25% |
| treatment before accident | 100% | 25% |
| injury answer consistent | 50% | 0% |
| policyholder at fault | 25% | 50% |
| full red-flag review | 0% | 0% |

**Error analysis** of the 9 agent misses: 6 were correct answers flagged by an over-strict verifier (quotes joining two excerpts with `...`, page numbers in the answer counted as unbacked numbers); fixed with tests. The real weakness was the open red-flag review: the agent stopped early and said "None found". Fix: an explicit checklist in the instructions (dates, injuries vs medical bills, vehicle on every estimate, treatment date). The re-run of the improved agent stopped at 15/40 when the free tier's 500 requests/day limit was reached; to be completed.

### LangChain / LangGraph implementation (`qa/lc.py`)

The same pipeline built from LangChain parts; the web app uses it by default (`COPILOT_IMPL=native` switches back to the plain Gemini SDK version):

- **`ClaimRetriever`**: a LangChain `BaseRetriever` over the hybrid search, restricted to one claim (and optionally document types); returns `Document`s with page, page type and OCR metadata.
- **Quick answer**: an LCEL chain, `RunnableParallel(docs=retriever, question=passthrough) | prompt | ChatGoogleGenerativeAI.with_structured_output(Answer)`; the retrieved documents are kept so code can verify the answer.
- **Deep agent**: `langchain.agents.create_agent` (runs on **LangGraph**) with LangChain `@tool`s (search, read_page, calculate, days_between) and a structured final answer via `ToolStrategy(FinalAnswer)`. The tools record pages opened and numbers computed, and the same verification code runs on the result.
- **Cost control**: an `InMemoryRateLimiter` for the free tier and a persistent JSONL LLM cache (`BaseCache`), so re-runs use no quota.
- Tests use fake LangChain chat models (scripted tool calls), so the chain and the LangGraph agent are tested without API calls. Live check: quick answer ($2,346.78, page 10) and the deep date check on claim 5 ("No: 07/18/2026 vs 07/17/2026", 2 verified citations).

### Demo web app (`app/`)

```bash
.venv/bin/python -m uvicorn app.server:app --port 8765    # then open http://localhost:8765
```

Pick a claim, browse its pages (each tagged with the sorter's type and a "scan" badge for OCR'd pages), and ask a question. The answer card shows the value, the cited page, the exact quote and the three verification checks; the cited page opens with the quote **highlighted**: on text pages via the PDF text layer, on scanned pages via Tesseract word boxes. A "What the model was given" panel lists the 5 retrieved chunks. The search engine loads at startup, so answers take about a second with Gemini 3.5 Flash-Lite.

**Model choice:** Gemini 3.8 Flash took 76-149 s per answer (long internal reasoning); 3.5 Flash-Lite answered the same questions correctly in under 1 s. For extracting values from retrieved text, the small fast model is the right tool.
