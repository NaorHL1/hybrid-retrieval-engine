# 🧠 Hybrid Retrieval Engine

A production-grade Retrieval-Augmented Generation (RAG) backend utilizing **Hybrid Search**. This engine combines the semantic understanding of Dense Vectors with the exact-keyword precision of Sparse Vectors (BM25), fused together using Reciprocal Rank Fusion (RRF) and ranked via a Neural Cross-Encoder.

---

## 🚀 Quick Start

### 1. Ingest the Data
Embed the dataset into the local LanceDB vector database. The dataset includes over 100 uniquely scraped, real-world intelligence and news reports regarding the FZ1073 incident.
```bash
uv run python ingest.py
```
> *Creates a local `vector_database` folder containing Apache Arrow tables and BGE embeddings.*

### 2. Start the Backend API (FastAPI)
Launch the multi-stage retrieval pipeline (BM25 -> RRF -> Cross-Encoder).
```bash
uv run uvicorn api:app --reload
```
> *API is now available at `http://127.0.0.1:8000`.*

### 3. Start the Frontend Dashboard (Streamlit)
Open a **new terminal window** and launch the UI:
```bash
uv run streamlit run app.py
```

---

## 🛠️ Technology Stack
* **Database:** LanceDB (Embedded, Serverless)
* **Embedding Model:** `BAAI/bge-small-en-v1.5`
* **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Fast local CPU inference)
* **Backend:** FastAPI (Python)
* **Frontend:** Streamlit
* **Package Management:** `uv`

---

## 📐 Architecture Decision Records (ADRs)

We followed strict architectural planning to ensure enterprise-grade scalability and privacy.

### ADR 001: Vector Database Selection
* **Decision:** `LanceDB` 
* **Why:** Serverless and embedded. Runs completely inside the application process (Zero DevOps overhead, no Docker required) and persists data locally. Most importantly, it provides **native support for Hybrid Search and RRF fusion out-of-the-box**.
* **Rejected:** *PostgreSQL+pgvector* (too much DevOps overhead), *Pinecone* (vendor lock-in, data privacy concerns for air-gapped systems).

### ADR 002: Local Embedding Model
* **Decision:** `BAAI/bge-small-en-v1.5`
* **Why:** Provides the perfect balance of retrieval quality, fast indexing speed, and low memory usage on local CPUs. Natively designed to support multi-vector and hybrid paradigms.
* **Rejected:** *OpenAI* (API costs, privacy concerns), *Qwen3-8B* (too large for standard hardware without GPU).

### ADR 003: Multi-Stage Reranking Pipeline
* **Decision:** `RRF + Local Listwise Cross-Encoder`
* **Why:** To achieve both massive scale and precision, we use a tiered approach:
  1. Retrieve top 1,000 candidates via Dense and Sparse searches.
  2. Fuse them instantly using **RRF** (Stage 1).
  3. Pass only the Top 15 candidates through a local **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) for precision reranking (Stage 2). 
* **Rejected:** *RRF Only* (lacks contextual understanding), *LLM-as-a-Judge* (unacceptable latency).

---

## ✈️ Dataset: Flydubai Flight 1073
To rigorously test our Hybrid RAG engine, we utilize a scraped dataset based on the recent September 2026 hijacking attempt of Flydubai Flight 1073 (Dubai to Tel Aviv). 

This domain perfectly illustrates the necessity of Hybrid Search:
* **Dense Search** captures semantic concepts *(e.g., "co-pilot attempting to crash the plane")*.
* **Sparse Search (BM25)** captures exact identifiers crucial for investigators *(e.g., "Hamam al-Hammami", "squawk 7500", "Boeing 737 MAX 8")*.
