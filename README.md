# 🧠 Hybrid Retrieval Engine (LanceDB)

A production-grade Retrieval-Augmented Generation (RAG) backend utilizing Hybrid Search. This engine combines the semantic understanding of Dense Vectors with the exact-keyword precision of Sparse Vectors (BM25), fused together using Reciprocal Rank Fusion (RRF).

## 📐 Architecture Decision Records (ADRs)

### 📄 ADR 001: Vector Database and Search Engine Selection
**Status:** `🟢 Accepted` | **Date:** October 2026

* **Context:** To implement a robust Hybrid Search architecture, we need a system capable of storing both dense vectors (for semantic search) and full-text documents (for sparse BM25 search). Furthermore, we need to fuse these results using RRF. The solution must be scalable, maintainable, and cost-effective for a medium-scale enterprise deployment.
* **Alternatives Considered:** 
  1. **PostgreSQL + pgvector:** Extremely robust and Enterprise-grade, but requires significant operational overhead (Docker, Network configuration) and complex SQL to manually fuse BM25 and Vector results.
  2. **Dedicated Cloud DB (Pinecone / Qdrant):** Highly scalable, but introduces vendor lock-in, external API latency, and data privacy concerns (CISO objections regarding sensitive data leaving the local perimeter).
  3. **Local In-Memory (FAISS + Python BM25):** Fast for prototyping, but lacks persistence. Data is lost upon server restart, making it unsuitable for production.
* **Decision:** We chose **LanceDB**.
* **Reasoning:** LanceDB is a serverless, embedded vector database. It runs completely inside the application process (Zero DevOps overhead, no Docker required) and persists data locally to the disk using the Apache Arrow format. Most importantly, it provides **native support for Hybrid Search and RRF fusion out-of-the-box** (utilizing the Rust-based Tantivy engine for full-text search). This grants us the persistence of a real database with the simplicity and speed of an in-memory Python library.

### 📄 ADR 002: Local Embedding Model Selection
**Status:** `🟢 Accepted` | **Date:** October 2026

* **Context:** For the Dense Search component of our hybrid engine, we need an embedding model to convert text into vectors. While cloud APIs (like OpenAI or Cohere) are viable, we proactively chose to utilize a local model via `sentence-transformers`. This strategic choice provides enhanced data privacy (keeping data on-premise), eliminates API costs, and avoids vendor lock-in, all while maintaining high retrieval accuracy on local hardware.
* **Alternatives Considered:** 
  1. **Qwen3-Embedding-8B:** Extremely accurate but too large (7.57B parameters) for standard local hardware without significant GPU VRAM.
  2. **EmbeddingGemma:** Highly efficient and low VRAM footprint, but restricted under Gemma Terms rather than a pure open-source MIT/Apache license.
  3. **BAAI/BGE-M3 / BGE-small:** Highly optimized for Dense, Sparse, and Multi-Vector retrieval. Open-source (MIT), exceptionally fast on local CPUs, and purpose-built for Hybrid search tasks.
* **Decision:** We chose **BAAI/BGE-M3** (or its lightweight variant `bge-small-en-v1.5`) via `sentence-transformers`.
* **Reasoning:** Based on the 2026 benchmark guide on [Best Local Embedding Models for RAG](https://atomic.chat/blog/guides/best-embedding-models-for-rag), BGE models offer the perfect balance of retrieval quality (nDCG@10), fast indexing speed, and low memory usage. Crucially, the BGE architecture is natively designed to support multi-vector and hybrid paradigms, making it the most architecturally sound choice for our local Hybrid LanceDB engine.


## ✈️ Dataset: Flydubai Flight 1073 Incident Reports
To rigorously test our Hybrid RAG engine, we utilize a mock dataset based on the recent September 2026 attempted hijacking of Flydubai Flight 1073 (Dubai to Tel Aviv). This domain perfectly illustrates the necessity of Hybrid Search:
* **Dense Search** captures semantic aviation security concepts (e.g., "a co-pilot attempting to crash the plane but being subdued by off-duty crew").
* **Sparse Search (BM25)** captures exact identifiers crucial for investigators (e.g., "Hamam al-Hammami", "squawk 7500", "Prince Sultan bin Abdulaziz Airport", "Boeing 737 MAX 8").
