# 🧠 Hybrid Retrieval Engine

A retrieval backend that combines dense vectors (semantic), BM25 (exact keywords), Reciprocal Rank Fusion (RRF) and a neural cross-encoder reranker, with an evaluation harness that measures each stage separately and a regression gate that runs under `pytest`.

The engineering point of this repo is the **measurement**: which stage helps, on which kind of query, how sure we are, and which of our own earlier claims did not survive measurement (see [Corrections](#-corrections-to-earlier-claims)).

---

## 🚀 Quick Start

### 1. Ingest the Data
Embed the dataset into the local LanceDB vector database: 155 scraped news/analysis excerpts about the FZ1073 incident plus 340 clearly synthetic distractor documents (see [Dataset](#️-dataset-flydubai-flight-1073)).
```bash
uv run python ingest.py
```
> *Creates a local `vector_database` folder (git-ignored). Re-run it whenever the files in `data/` change.*

### 2. Start the Backend API (FastAPI)
```bash
uv run uvicorn api:app --reload
```
> *Available at `http://127.0.0.1:8000`. Endpoints: `GET /search?query=` (dense only) and `GET /hybrid-search?query=` (dense + BM25 + RRF + cross-encoder).*

### 3. Start the Frontend Dashboard (Streamlit)
In a **new terminal**:
```bash
uv run streamlit run app.py
```

### 4. Tests and evaluation
```bash
uv run pytest                          # everything; a few minutes on CPU (needs the built DB and models)
uv run pytest test_retrieval.py eval/test_metrics.py eval/test_gate.py   # fast, no DB or models
uv run python -m eval.run              # score all stages on 45 queries, print tables, write eval/results.json
uv run python -m eval.run --write-baseline   # freeze eval/baseline.json (deliberate; review the diff)
uv run pytest eval/test_regression.py  # fail if quality dropped more than 0.05 vs the baseline (~100 s)
```
`benchmark.py` is the original 20-query script with single-id labels; it still runs but is superseded by `eval/` (see [Corrections](#-corrections-to-earlier-claims)).

---

## 🛠️ Technology Stack
* **Database:** LanceDB (embedded, serverless)
* **Embedding model:** `BAAI/bge-small-en-v1.5`
* **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local CPU inference)
* **Backend:** FastAPI · **Frontend:** Streamlit · **Packages:** `uv`

### Code layout
* `retrieval.py`: the stage functions (`dense`, `bm25`, `hybrid_rrf`, `rerank`). The API and the eval both call these, so the eval measures the code that serves requests.
* `api.py`: FastAPI app. One DB handle, opened once; the cross-encoder runs in an executor behind a semaphore (`CE_CONCURRENCY`, default 1) so a slow rerank does not block the event loop.
* `eval/`: queries, metrics, runner, regression gate, and `FINDINGS.md` (running log of what we measured and retracted).

---

## 📐 Architecture Decision Records (ADRs)

### ADR 001: Vector Database Selection
* **Decision:** `LanceDB`
* **Why:** Embedded (no DevOps, no Docker), persists locally, and has native hybrid search and RRF.
* **Rejected:** *PostgreSQL + pgvector* (more operational overhead), *Pinecone* (vendor lock-in, data-privacy concerns for air-gapped use).

### ADR 002: Local Embedding Model
* **Decision:** `BAAI/bge-small-en-v1.5`
* **Why:** Good quality for its size, fast indexing, low memory on CPU.
* **Rejected:** *OpenAI* (API cost, privacy), *Qwen3-8B* (too large without a GPU).

### ADR 003: Multi-Stage Pipeline
* **Decision:** dense + BM25 → RRF → local cross-encoder.
* **Flow:** fuse both lists over the whole corpus with RRF (limit 1000), keep the top 50, rerank those 50 with the cross-encoder, return the top 5.
* **Final order uses the cross-encoder score only.** The BM25 and RRF scores influence only *which 50 documents* are reranked.
* **Rejected:** *RRF only* (no contextual understanding), *LLM-as-a-judge* (latency).
* **Open question this repo answers with data:** whether each stage earns its place (see Results).

---

## ✈️ Dataset: Flydubai Flight 1073

Real documents: 155 excerpts about the September 2026 Flydubai Flight 1073 hijacking attempt, scraped from several outlets. Properties that matter for evaluation:

* **Near-duplicates are common.** Different outlets report the same fact, so several documents can answer one question (for example two documents both say a dentist treated the pilot). Single-id labels would punish correct answers, which is why scoring has a *strict* and a *lenient* variant.
* **Scraping artifacts.** Some words are glued together (`staffEyal Zamirheld`), so BM25 cannot match `Eyal Zamir` while dense search can. At least 16% of documents contain such joins (a lower bound).

Synthetic documents (ids `doc_1001` and up; not news):
* 304 topical distractors on unrelated subjects.
* 36 **look-alike identifier distractors** (`A6-FKE`, `+971 54 232 3579`, `squawk 7600`, ...), written to contain near-misses of real identifiers but never a full target string and never an answer to a query. A short version (about 130 characters) and a length-matched version (about 409 characters, matching the real targets) exist; the default is length-matched (`LOOKALIKE_FILE` selects the other). Why both exist is explained under Results.

---

## 📊 Evaluation

### Method
* **45 queries** in `eval/queries.jsonl`: 15 *semantic* (paraphrased questions), 15 *identifier* (registrations, phone numbers, names, times, laws), 15 *mixed* (an identifier shared by several documents plus free words that disambiguate). Each has an `expected_id` and an `also_relevant` list; all were checked by hand against the full text of the documents.
* **Strict** counts only `expected_id`; **lenient** also counts `also_relevant` duplicates. Lenient is the headline.
* **Four configurations:** `dense`, `bm25`, `rrf` (fusion only), `rrf+ce` (the production pipeline).
* **Metrics:** Hit@1, Hit@3, MRR, and candidate Recall@50 (does the right document reach the cross-encoder at all). 95% bootstrap intervals (1000 resamples, fixed seed), and a *paired* bootstrap for differences between neighboring stages.
* The relevant-document pools used for labeling came from dense and BM25 only (not the cross-encoder), to avoid biasing toward the reranker.

### Results (45 queries, 495 documents)

| config | Hit@1 strict | Hit@3 strict | Hit@1 lenient | Hit@3 lenient [95% CI] | MRR lenient |
|---|---|---|---|---|---|
| dense | 0.42 | 0.67 | 0.47 | 0.73 [0.60, 0.84] | 0.62 |
| bm25 | 0.62 | 0.76 | 0.69 | 0.80 [0.67, 0.91] | 0.75 |
| rrf | 0.51 | 0.78 | 0.56 | 0.82 [0.71, 0.93] | 0.70 |
| **rrf+ce** | **0.69** | **0.84** | **0.76** | **0.89** [0.80, 0.98] | **0.83** |

Candidate Recall@50 (rrf top-50 reaches the cross-encoder): 0.96 strict, 0.98 lenient.

Hit@1 / Hit@3 per bucket (lenient):

| config | identifier | mixed | semantic |
|---|---|---|---|
| dense | 0.27 / 0.67 | 0.73 / 0.93 | 0.40 / 0.60 |
| bm25 | 0.87 / 0.87 | 0.80 / 0.87 | 0.40 / 0.67 |
| rrf | 0.40 / 0.87 | 0.87 / 0.87 | 0.40 / 0.73 |
| rrf+ce | 0.67 / 0.87 | 0.87 / 0.87 | 0.73 / 0.93 |

### What the data supports
Findings below use paired 95% intervals on the same 45 queries. Where a difference "excludes zero" it is unlikely to be query-sampling noise; with this many comparisons a few will appear by chance, so read the *pattern*, not single cells.

* **Hit@3 differences between stages are not statistically distinguishable at n=45.** All three neighboring-stage intervals for overall Hit@3 include zero (for example `rrf+ce − rrf = +0.07`, 95% CI `[−0.02, +0.16]`). The point estimates all point in the expected direction, but this query set cannot separate them.
* **Rank quality (Hit@1, MRR) separates the stages more clearly.** Overall lenient: BM25 beats dense (+0.22 Hit@1) and the cross-encoder improves on RRF (+0.20 Hit@1); both intervals exclude zero.
* **The cross-encoder helps most on semantic queries** (lenient Hit@1 0.40 → 0.73, paired `+0.33`, interval excludes zero).
* **Dense search is weakest on exact identifiers** (identifier Hit@1 0.27 vs BM25 0.87).
* **RRF fusion lowered top-1 accuracy on identifiers compared to BM25 alone** (0.40 vs 0.87, paired `−0.47`, interval excludes zero) once look-alike identifiers were in the corpus; the cross-encoder then recovered part of it (0.67). Fusion did not hurt on semantic or mixed queries. Without the look-alikes, RRF matched BM25 on identifiers (0.87 vs 0.87).

### The look-alike experiment (and the confound we found)
Identifier bucket, strict Hit@1 / Hit@3, n=15:

| config | no look-alikes | short look-alikes | length-matched (default) |
|---|---|---|---|
| dense | 0.73 / 0.80 | 0.20 / 0.60 | 0.27 / 0.67 |
| bm25 | 0.87 / 0.93 | 0.40 / 0.73 | 0.87 / 0.87 |
| rrf | 0.87 / 0.87 | 0.33 / 0.73 | 0.40 / 0.87 |
| rrf+ce | 1.00 / 1.00 | 0.67 / 0.87 | 0.67 / 0.87 |

The first version of the look-alikes was about 3x shorter than the real targets, and BM25's length normalization favors short documents, so part of its drop was an artifact of how we wrote the data. Matching the length restored BM25 completely, while dense stayed hurt. We report both so the effect of length is visible.

### Cost
* Dense search (including embedding the query) about 50 ms and BM25 about 14 ms per query at p50; the cross-encoder dominates the pipeline at about 1.3 to 1.7 s p50 for 50 candidates on this CPU-only machine (several eval runs, using `batch_size=8`).
* These are single-machine timings that vary a lot between runs (we saw 9 to 14 s per query for the same setting under load before a long document was removed and the batch size reduced), so treat them as indicative, not as a benchmark.

### Corrections to earlier claims
Earlier versions of this README said pure vector search scored 100% and hybrid 95% on 20 queries, explained by a "Lexical Penalty" in which BM25 gave the correct document a score of 0 and dragged it out of the top 3. We retired that story:

* **The mechanism is not implemented.** The final ranking sorts by cross-encoder score only; BM25 and RRF scores never reach the final order. They only decide which 50 documents the cross-encoder sees.
* **The miss was a labeling artifact plus a reranker preference, not BM25.** For the query "Who was the medical professional that treated the wounded pilot?", the labeled document (`doc_0010`) was inside the 50 candidates (RRF rank 5). The cross-encoder ranked a different document first (`doc_0142`, score 0.99) that also states a dentist treated the pilot. Counted as a hit under lenient labels.
* **The old numbers do not reproduce.** They came from a 116-document corpus with single-id labels. On the current corpus `benchmark.py` gives 65% (vector) and 80% (hybrid) on its 20 queries; that is not comparable to the old figures.
* The earlier README also asserted that pure vector search "hallucinates on exact IDs" at scale. The data is more nuanced: on the corpus without look-alike identifiers, dense ranked the target first for 11 of 15 identifier queries (Hit@1 0.73), and its clearest failures were time strings (`10:45 a.m. GST` at rank 173 of 460 in one early measurement). It got clearly worse once look-alike identifiers were added (Hit@1 0.27). That is evidence for BM25's value on identifiers, measured on this small corpus, not a general claim about scale.

### Limitations
* **n=45 (15 per bucket).** One query is worth 0.022 of overall Hit@3 and 0.067 of a bucket. Differences smaller than that are not interpretable.
* **Queries and distractors were drafted by one author** (an AI assistant) from the target documents, then verified by a human. Shared vocabulary with the targets can inflate scores; a set written blind by someone else would test this.
* **One corpus, one incident,** with near-duplicates and scraping noise, plus synthetic distractors.
* **Defaults only.** The FTS tokenizer splits hyphenated identifiers (`A6-FKF` becomes `a6` and `fkf`, matched with OR semantics), and the cross-encoder truncates input at 512 tokens. Neither was tuned.
* **Latency was measured on one CPU machine.**

### Regression gate
`eval/baseline.json` stores two gated overall-lenient numbers (`rrf+ce` Hit@3 = 0.8889, candidate Recall@50 = 0.9778) and a fingerprint of the corpus, queries and settings. `pytest eval/test_regression.py` fails if either number drops by more than 0.05 (three lost queries fail, two pass), or if the fingerprint differs, which means the baseline is stale and must be refrozen on purpose. Both failure modes were verified by breaking the code deliberately.

---

## 🔬 Reciprocal Rank Fusion (RRF)
RRF merges the dense and BM25 rankings without calibrating their scores:
`RRF_score(doc) = Σ 1 / (k + rank_i)`, with `k = 60`.
A document ranked well by both retrievers beats one ranked first by only one. `retrieval.rrf_fuse` reimplements this for tracing, and a test checks that it produces the same order as LanceDB's `RRFReranker` on 10 real queries.
