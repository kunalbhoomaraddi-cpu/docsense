# DocSense

**Ask questions about your documents and get answers that cite their source, or an honest "I couldn't find this".**

**Live demo: [docsense-6bcu.onrender.com](https://docsense-6bcu.onrender.com)** (free hosting: the first visit can take ~50 s to wake up)

![CI](https://github.com/kunalbhoomaraddi-cpu/docsense/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-22%20passing-2ea44f)
![Made-up answers](https://img.shields.io/badge/made--up%20answers%20on%20eval-0-2ea44f)
![Gemini eval](https://img.shields.io/badge/live%20eval%20with%20Gemini-43%2F43-2ea44f)

DocSense is a retrieval-augmented generation (RAG) app. Upload PDFs, Markdown or text files, ask a question, and it answers from the documents only. Every sentence links to the exact file and section it came from. When the documents don't contain the answer, it says so instead of guessing.

It ships with an **evaluation suite**: 43 labelled questions (including 10 that are deliberately *not* answerable), and a script that measures retrieval quality, refusal behaviour and answer accuracy. Every design choice below is backed by a number from that suite.

| Answer with a citation | Refusing when the answer isn't there |
|---|---|
| ![answer](docs/answer.png) | ![refusal](docs/refusal.png) |

## Results

From [`evals/results.md`](evals/results.md): 43 labelled questions, 10 of them deliberately unanswerable.

| Metric | Offline (no API key) | **With Gemini (live app)** |
|---|---|---|
| Right section ranked first (Hit@1) | 85% | – |
| Answer contains the expected fact | 85% | **100%** |
| Answer cites the right section | 85% | **100%** |
| Unanswerable questions correctly refused | 100% | **100%** |
| Made-up answers | **0** | **0** |
| Median answer time | <10 ms | ~1.1 s |

The offline numbers are generated in CI on every push (`python -m evals.run_evals`). The Gemini column is a run
against the deployed app on 2026-11-20 (raw results in `evals/results_gemini_live.json`). A perfect score on 43
questions says the eval set is now too easy, not that the system is perfect; harder questions are the next step.

## How it works

```
upload ──► split into sections/pages ──► chunk (≤600 chars, 1-sentence overlap)
                                                   │
question ──► hybrid retrieval (BM25 + TF-IDF [+ Gemini embeddings]) ──► top 4 chunks
                                                   │
                         confidence = ½ relevance + ½ query coverage
                                                   │
                     below 0.38? ──► "I couldn't find this in the documents."
                                                   │
                     answer from numbered sources ──► check citations ──► cited answer
```

1. **Ingest** (`docsense/ingest.py`): Markdown is split by heading and PDFs by page, so every chunk has exactly one citation (`travel-and-expenses.md › Hotel limits`, `handbook.pdf p.4`).
2. **Retrieve** (`docsense/retrieval.py`): BM25 (written from scratch) and TF-IDF cosine similarity, combined with Reciprocal Rank Fusion. With a `GEMINI_API_KEY`, Gemini embeddings join the fusion.
3. **Decide** (`docsense/pipeline.py`): before calling any LLM, DocSense checks whether the best passage really covers the question. *Query coverage* is the share of the question's meaning (IDF-weighted words) that appears in the passage.
4. **Answer** (`docsense/answer.py`):
   - **Gemini** (when a key is set) answers only from numbered sources. Citations that point at sources that don't exist are removed, and an answer with no valid citation is refused.
   - **Extractive** (no key needed) picks the best-matching sentences and cites them. It cannot invent text, which makes it a useful baseline.

## Decisions I made, and why

- **Refuse before calling the LLM.** Retrieval relevance alone could not separate answerable from unanswerable questions: "dress code" still retrieves *something*. Adding query coverage took correct refusals **from 50% to 100%** and cut made-up answers **from 5 to 0** on the eval set. The threshold (0.38) is the value that best balances the two in the sweep table in `evals/results.md`.
- **Chunks never cross a section or page.** One chunk = one citation, so a citation can never point at the wrong place.
- **Hybrid retrieval as the default, even though TF-IDF alone scored slightly higher offline** (Hit@1 88% vs 85%). Hybrid is what lets Gemini embeddings plug in when a key is available, and RRF makes adding a retriever a one-line change. The offline numbers are reported as they are.
- **Chunk size barely mattered** (300 / 600 / 1000 chars all scored the same), because policy sections are short. I kept 600 as a middle ground for longer PDFs.
- **The AI can be slow or down; the app can't.** The first live run hit a 60-second timeout from Gemini. Every Gemini call now retries, switches to a lighter fallback model (`gemini-3.5-flash-lite`) if the main one is slow, and if Gemini is completely unavailable the app returns its offline result instead of an error. Tests simulate timeouts and outages.
- **No vector database.** For a few hundred chunks, in-memory NumPy/scikit-learn is faster and simpler. A store like pgvector or Qdrant becomes worth it at tens of thousands of chunks.

## Known limitations

- **Offline mode has vocabulary gaps.** Without an API key, the 5 misses are all questions that use different words from the documents ("vacation" vs "annual leave", "fathers" vs "paternity", "Dubai" vs "outside India"). With Gemini embeddings switched on, all 5 are answered correctly (see Results).
- **With embeddings, the pre-LLM refusal check does less.** Embedding scores run higher, so the confidence threshold alone refused 3 of 10 unanswerable questions in the live run; the LLM's NOT_FOUND rule and citation check caught the other 7. The threshold should be re-tuned separately for the Gemini configuration.
- Uploaded documents are shared by all visitors to one running instance (fine for a demo, not for multiple users), and scanned PDFs need OCR first.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn docsense.api:app --reload        # open http://localhost:8000
```

Optional: copy `.env.example` to `.env` and add a free [Gemini API key](https://aistudio.google.com/apikey) to switch on LLM answers and embeddings.

```bash
pytest -q                                # 22 tests, no network needed
python -m evals.run_evals                # regenerate evals/results.md
GEMINI_API_KEY=... python -m evals.run_evals --llm   # also score Gemini answers
```

**API:** `POST /ask {"question": "..."}` · `POST /documents` (file upload) · `DELETE /documents` (reset) · `GET /health`. Interactive docs at `/docs`.

**Deploy:** the repo includes a `Dockerfile` and `render.yaml`. On [Render](https://render.com), choose *New → Blueprint*, select this repo, and add `GEMINI_API_KEY` if you want LLM answers.

## Project structure

```
docsense/        ingest, retrieval, pipeline, answer, llm client, FastAPI app + static UI
data/sample/     4 sample policy documents (fictional company)
evals/           labelled dataset, eval runner, results.md
tests/           unit + API tests (Gemini mocked)
```

---
Built by [Kunal Bhoomaraddi](https://github.com/kunalbhoomaraddi-cpu) · MIT License
