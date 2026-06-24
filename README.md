<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Samsung_Logo.svg/2560px-Samsung_Logo.svg.png" height="40" alt="Samsung"/>
&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Samsung-PRISM-1428A0?style=for-the-badge&logoColor=white"/>

# Code RAG using Knowledge Graph and Vector Database

**Samsung PRISM Work-let · End Review · June 2026**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*A hybrid GraphRAG system that lets developers query large Java codebases in natural language — combining AST-based Knowledge Graphs with Vector Database retrieval for structurally accurate, contextually grounded answers.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Sub-Projects](#sub-projects)
  - [Hybrid RAG — CodeGraph (Streamlit)](#1-hybrid-rag--codegraph-streamlit-dashboard)
  - [KG and Others — Benchmarking Suite](#2-kg-and-others--rag-benchmarking-suite)
  - [Vector DB Code RAG — CodeRAG AI (React + FastAPI)](#3-vector-db-code-rag--coderag-ai-react--fastapi)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Patentable Innovations](#patentable-innovations)
- [Research & Publication](#research--publication)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

Traditional Retrieval-Augmented Generation (RAG) systems treat source code like plain text — splitting it at arbitrary line boundaries and indexing fragments by semantic similarity. This approach fundamentally fails for structural queries: *"Trace the request path from the REST controller to the database"* cannot be answered by keyword proximity alone. Code is a **graph**, not a document.

This project builds, benchmarks, and compares three distinct Code RAG architectures:

| Architecture | Description | Key Technology |
|---|---|---|
| **No-Graph Baseline** | Vector-only semantic search | Chroma DB / Neo4j vector index |
| **DKB GraphRAG** | Deterministic AST parsing + graph traversal | Tree-sitter + Neo4j + NetworkX |
| **LLM-KB GraphRAG** | LLM-driven relation extraction + graph traversal | Gemini + NetworkX |

All three systems were evaluated across **three production-grade open-source Java codebases** on **15 standardized engineering questions**, generating a fully reproducible comparative benchmark.

**Key finding:** The Hybrid DKB GraphRAG system achieves the lowest query latency (~10.6s), requires only ~10–15% additional build overhead over the vector baseline, and provides structurally grounded answers that pure vector search cannot produce. Tree-sitter graph extraction is **64× faster** than LLM-based extraction, making it the only viable approach for large-scale production indexing.

---

## Architecture

The system is designed as a **three-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: INGESTION PIPELINE                       │
│                       (Session-based)                                │
│                                                                      │
│  GitHub URL / Local Path                                             │
│         │                                                            │
│         ▼                                                            │
│  File Traversal & Filtering (.java, .py, .js)                        │
│         │                                                            │
│    ┌────┴────────────────────────────┐                               │
│    ▼                                 ▼                               │
│  Code Chunking Engine          Tree-sitter AST Parser                │
│  (AST leaf boundaries)               │                               │
│         │                    ┌───────┴──────────┐                   │
│         ▼                    ▼                  ▼                    │
│  Generate Embeddings    Extract Entities   Extract Relationships      │
│  (OpenAI / Google /     (Classes, Methods, (CALLS, INHERITS,         │
│   Jina / Gemini)         Packages)          IMPORTS)                 │
└──────────────┬──────────────────────┬───────────────────────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STAGE 2: HYBRID STORAGE LAYER                          │
│                  (Unified Neo4j Instance)                             │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Neo4j Database                          │   │
│   │  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│   │  │ Vector Embeddings│  │  CodeNodes   │  │ Relationships │  │   │
│   │  │ (Indexed for     │  │ (AST Props)  │  │ (CALLS,       │  │   │
│   │  │  Similarity)     │  │              │  │  IMPORTS,     │  │   │
│   │  └─────────────────┘  └──────────────┘  │  INHERITS)    │  │   │
│   │                                          └───────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 3: RETRIEVAL & GENERATION PIPELINE                │
│                         (On-line / Query-time)                       │
│                                                                      │
│  User Query (Natural Language)                                       │
│       │                                                              │
│       ▼                                                              │
│  Query Enhancement LLM  ──►  Embed Enhanced Query                   │
│                                       │                              │
│                                       ▼                              │
│                          ANN Vector Search in Neo4j                  │
│                          (Cosine Similarity)                         │
│                                       │                              │
│                                       ▼                              │
│                          Retrieve Top-K Code Chunks                  │
│                                       │                              │
│                                       ▼                              │
│                    Bridge Layer: Map Chunks → AST Entities           │
│                    (via REPRESENTS relationships)                     │
│                                       │                              │
│                                       ▼                              │
│                    Graph Traversal Engine                            │
│                    (Fetch: Calling Context / Imports / Siblings)     │
│                                       │                              │
│                                       ▼                              │
│                    Assemble Context                                  │
│                    (Code Chunks + Graph Topology)                    │
│                                       │                              │
│                                       ▼                              │
│                    LLM Generator (Gemini / GPT-4o)                  │
│                                       │                              │
│                                       ▼                              │
│                    Grounded Answer with Source Citations             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
CodeRAG/
│
├── hybrid rag/                     # Sub-project 1: CodeGraph Streamlit Dashboard
│   ├── main.py                     #   Entry point — Streamlit app
│   ├── storage.py                  #   Chunked Neo4j ingestion (UNWIND Cypher)
│   ├── graph_builder.py            #   Tree-sitter AST → Neo4j graph construction
│   ├── embeddings.py               #   Parallel thread-pool embedding generation
│   ├── retrieval.py                #   Hybrid vector + graph retrieval pipeline
│   └── requirements.txt
│
├── kg and others/                  # Sub-project 2: Benchmarking & Evaluation Suite
│   ├── benchmark.py                #   135-run evaluation harness (3×3×15)
│   ├── dkb_rag.py                  #   DKB (Tree-sitter) GraphRAG implementation
│   ├── llm_kb_rag.py               #   LLM-KB GraphRAG implementation
│   ├── baseline_rag.py             #   No-Graph vector baseline implementation
│   ├── questions.json              #   15 standardized engineering questions
│   ├── json/                       #   Evaluation output — answers + latency metrics
│   ├── png/                        #   Dependency tree topology visualizations
│   └── logs/                       #   Execution logs from pipeline runs
│
├── vector db code rag/             # Sub-project 3: CodeRAG AI Full-Stack App
│   ├── server_v1/                  #   FastAPI backend
│   │   ├── main.py                 #     WebSocket + REST API
│   │   ├── session.py              #     Session-isolated graph management
│   │   ├── storage.py              #     Neo4j + Redis integration
│   │   └── requirements.txt
│   └── client/                     #   React SPA frontend
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   └── hooks/
│       └── package.json
│
├── .gitignore
└── README.md
```

---

## Sub-Projects

### 1. Hybrid RAG — CodeGraph (Streamlit Dashboard)

**Location:** `hybrid rag/`  
**Interface:** Streamlit desktop dashboard  
**Stack:** Python · Streamlit · Neo4j · OpenAI GPT-4o · Tree-sitter

The flagship implementation of the Deterministic Knowledge-Base GraphRAG (DKB). A single-user analytics dashboard for ingesting a codebase, inspecting the resulting knowledge graph, and querying it interactively.

**Features:**
- One-click codebase ingestion from a GitHub URL or local folder path
- Real-time database statistics (node counts, relationship counts, index coverage)
- Interactive chat interface for natural language code queries
- Chunked `UNWIND` Cypher ingestion — zero out-of-memory (OOM) failures on large codebases
- Parallelized embedding generation via thread pool for fast indexing

**Run:**
```bash
cd "hybrid rag"
pip install -r requirements.txt
streamlit run main.py
```

---

### 2. KG and Others — RAG Benchmarking Suite

**Location:** `kg and others/`  
**Interface:** CLI / JSON outputs  
**Stack:** Python · Chroma DB · Neo4j · Google Gemini · NetworkX · Tree-sitter · matplotlib

A headless benchmarking harness that runs all three RAG architectures against 15 standardized engineering questions across all three codebases, recording build times, graph construction overhead, and query latencies. This sub-project produces the quantitative evidence underlying all experimental claims in the project report.

**Codebases evaluated:**

| Codebase | Scale | Domain |
|---|---|---|
| [ThingsBoard](https://github.com/thingsboard/thingsboard) | Large | IoT platform |
| [OpenMRS Core](https://github.com/openmrs/openmrs-core) | Medium | Medical records |
| [Shopizer](https://github.com/shopizer-ecommerce/shopizer) | Small | E-commerce |

**Architectures evaluated:**

| ID | Method | Graph Construction | Vector Store |
|---|---|---|---|
| `no-graph` | Baseline | None | Chroma DB / Neo4j vector |
| `dkb` | Deterministic KG | Tree-sitter AST | Neo4j |
| `llm-kb` | LLM-extracted KG | Gemini LLM calls | NetworkX + Neo4j |

**Run benchmarks:**
```bash
cd "kg and others"
pip install -r requirements.txt
python benchmark.py --codebase thingsboard --arch all
```

**Output:** Results are saved to `json/`, topology visualizations to `png/`, and execution logs to `logs/`.

---

### 3. Vector DB Code RAG — CodeRAG AI (React + FastAPI)

**Location:** `vector db code rag/`  
**Interface:** React SPA web application  
**Stack:** React · FastAPI · Neo4j · Redis · Google Gemini · WebSockets

A production-oriented, multi-user web application for codebase querying. Supports concurrent sessions with full data isolation — multiple users can index different repositories simultaneously without interference.

**Features:**
- Session-isolated graph ingestion: each user's data is scoped to a unique session identifier in Neo4j, preventing cross-user data leakage
- Live git repository cloning directly from the UI
- WebSocket-based real-time indexing progress and query streaming
- Multi-user session management via Redis
- REST API + WebSocket API for integration with external tools

**Run backend:**
```bash
cd "vector db code rag/server_v1"
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Run frontend:**
```bash
cd "vector db code rag/client"
npm install
npm run dev
```

---

## Key Results

### Build Time — Vector DB & Graph Construction

| Codebase | Method | Vector DB Build (s) | Graph Build (s) | Total (s) |
|---|---|---|---|---|
| **ThingsBoard** (Large) | No-Graph Baseline | 123.61 | 0.00 | **123.61** |
| | DKB (Tree-sitter) | 129.78 | **13.77** | **143.55** |
| | LLM-KB (LLM Extractor) | 95.51 | **883.74** | **979.25** |
| **OpenMRS Core** (Medium) | No-Graph Baseline | 42.20 | 0.00 | **42.20** |
| | DKB (Tree-sitter) | 42.82 | 5.60 | **48.42** |
| | LLM-KB (LLM Extractor) | 29.12 | 222.17 | **251.29** |
| **Shopizer** (Small) | No-Graph Baseline | 18.41 | 0.00 | **18.41** |
| | DKB (Tree-sitter) | 19.28 | 2.81 | **22.09** |
| | LLM-KB (LLM Extractor) | 14.95 | 200.14 | **215.09** |

> **Key Insight:** DKB (Tree-sitter) adds only ~10–15% overhead over the No-Graph baseline. LLM-KB adds a **~64× increase** in graph build time — making it unsuitable for production indexing of large codebases.

---

### Query Performance

| Architecture | Avg. Query Latency | Notes |
|---|---|---|
| **DKB GraphRAG** (Tree-sitter) | **~10.6s** ✅ Fastest | Graph-based context is precise — less noise sent to LLM |
| **No-Graph Baseline** | ~11.3s | Competitive but lacks structural context |
| **LLM-KB** (LLM Extractor) | ~12.2s | Dense context window → higher LLM inference time |

---

### System Comparison

| Criterion | Vector DB Only | Knowledge Graph Only | **Hybrid (Recommended)** |
|---|---|---|---|
| Build speed | ✅ Fast (18–123s) | ⚠️ Variable | ✅ Fast (+10–15%) |
| Natural language queries | ✅ Good | ❌ Fragile | ✅ Excellent |
| Structural queries | ❌ Poor | ✅ Excellent | ✅ Excellent |
| Dependency tracing | ❌ No | ✅ Yes | ✅ Yes |
| NL → code translation | ✅ Handles well | ❌ Fails on ambiguous terms | ✅ Handles well |
| Query latency | ~11.3s | N/A | **~10.6s** |
| Infrastructure complexity | Low | Medium | Medium (Neo4j + vector) |

---

## Dataset

Evaluation data is organized in three folders within `kg and others/`:

| Folder | Contents | Format | Count |
|---|---|---|---|
| `json/` | Structured evaluation results — generated answers, latency metrics | JSON | 3 codebases × 3 architectures × 15 questions = **135 runs** |
| `png/` | Dependency tree topology visualizations | PNG | 6 images |
| `logs/` | Full execution logs from pipeline runs | TXT | 9 log files |

**Code volume (KLOC):**

```
Vector DB CodeRAG          1.2K lines
Knowledge Graph CodeRAG    1.2K lines
Hybrid CodeRAG             3.5K lines
Documentation / Frontend / Backend / JSON benchmarks   14.2K lines
─────────────────────────────────────────────────────
Total Workspace            20.1K lines
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Graph Database** | Neo4j | Stores code nodes, relationships, and vector indexes |
| **In-Memory Graph** | NetworkX | Graph construction and traversal during benchmarking |
| **Vector Store** | Chroma DB | Baseline vector-only RAG experiments |
| **AST Parser** | Tree-sitter | Deterministic extraction of code structure (nodes + edges) |
| **Embeddings** | OpenAI `text-embedding-3-large` (3072-dim) | High-fidelity code semantic embeddings |
| **Embeddings** | Google `text-embedding-004` (1024-dim) | Alternative embedding model |
| **Generator LLM** | OpenAI GPT-4o | Primary answer generation |
| **Generator LLM** | Google Gemini | Alternative / benchmarking LLM |
| **Cache / Sessions** | Redis | Multi-user session management and caching |
| **Backend API** | FastAPI + WebSockets | Real-time streaming responses and REST endpoints |
| **Frontend** | React (SPA) | Multi-user web interface |
| **Dashboard** | Streamlit | Analytics and single-user interactive interface |
| **Visualization** | matplotlib + NetworkX spring layout | Dependency tree topology rendering |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm (for the React client)
- A running [Neo4j](https://neo4j.com/download/) instance (v5+) with the `apoc` and vector index plugins enabled
- Redis (for the full-stack app)
- API keys: OpenAI and/or Google Gemini

### Clone the Repository

```bash
git clone https://github.com/adt-kmr/CodeRAG.git
cd CodeRAG
```

### Install Python Dependencies

Each sub-project has its own `requirements.txt`. Install for the sub-project you want to run:

```bash
# For the Streamlit dashboard (Hybrid RAG)
pip install -r "hybrid rag/requirements.txt"

# For the benchmarking suite
pip install -r "kg and others/requirements.txt"

# For the FastAPI backend
pip install -r "vector db code rag/server_v1/requirements.txt"
```

### Install Frontend Dependencies

```bash
cd "vector db code rag/client"
npm install
```

---

## Configuration

Create a `.env` file in the sub-project directory you are running. The following variables are used across sub-projects:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=AIza...

# Redis (for full-stack app only)
REDIS_URL=redis://localhost:6379

# Optional: Jina AI embeddings
JINA_API_KEY=jina_...
```

> **Note:** Neo4j must have the vector index plugin enabled. Run the following in Neo4j Browser to confirm:
> ```cypher
> CALL db.indexes() YIELD type WHERE type = 'VECTOR' RETURN count(*);
> ```

---

## Usage

### Streamlit Dashboard (Hybrid RAG)

```bash
cd "hybrid rag"
streamlit run main.py
```

1. Open `http://localhost:8501` in your browser.
2. Enter a GitHub URL (e.g., `https://github.com/thingsboard/thingsboard`) or a local folder path.
3. Click **Ingest** — the system will clone, parse, embed, and index the codebase.
4. Use the **Chat** interface to ask natural language questions:
   - *"Which classes implement AlarmDao?"*
   - *"Trace the request path from the REST controller to the database persistence layer."*
   - *"How does the authentication flow work?"*

---

### Benchmarking Suite

```bash
cd "kg and others"

# Run all architectures on all codebases
python benchmark.py --all

# Run a specific architecture on a specific codebase
python benchmark.py --codebase shopizer --arch dkb

# Available architectures: no-graph | dkb | llm-kb
# Available codebases: thingsboard | openmrs | shopizer
```

Results are written to `json/`, visualizations to `png/`, and logs to `logs/`.

---

### Full-Stack Web App (CodeRAG AI)

**Terminal 1 — Start the backend:**
```bash
cd "vector db code rag/server_v1"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the frontend:**
```bash
cd "vector db code rag/client"
npm run dev
```

Open `http://localhost:5173`. Multiple users can open separate sessions — each session maintains its own isolated graph and vector index scope within the shared Neo4j instance.

---

## Benchmarks

All 135 benchmark runs (3 codebases × 3 architectures × 15 questions) are reproducible from the `kg and others/` sub-project. The 15 standardized engineering questions are in `kg and others/questions.json` and cover:

- Direct class/method lookup (*"Find the class handling JWT token validation"*)
- Dependency tracing (*"Trace the request from the REST endpoint to database persistence"*)
- Architecture queries (*"Which classes implement the AlarmDao interface?"*)
- Cross-cutting concerns (*"How does the Spring security configuration work?"*)
- Performance-sensitive paths (*"Where is caching implemented in the service layer?"*)

Raw results are in `json/`, topology graphs in `png/`, and pipeline logs in `logs/`.

---

## Patentable Innovations

This project produced two novel technical contributions under evaluation for patent filing:

### Innovation 1: Session-Isolated Graph Ingestion and Vector Coexistence

A method for dynamically building **session-isolated, multi-tenant AST codebase dependency graphs** inside a single Neo4j database instance.

- Users index repositories on-the-fly under a temporary session identifier
- Relationships (`IMPORTS_FROM`, `CALLS`) are bounded strictly within the session scope
- A session-isolated vector search procedure (`db.index.vector.queryNodes` combined with Cypher session filtering) executes queries concurrently **without cross-user data leakage**

This enables enterprise multi-tenant deployment on shared Neo4j infrastructure without per-user database instances.

### Innovation 2: Deterministic Hybrid GraphRAG Expansion (AST-guided Retrieval)

An algorithm that uses local **Tree-sitter AST parsing** to map vector-retrieved code chunks back to their semantic graph nodes, then runs **deterministic graph traversals** to gather structural context — selectively pruning to fit within LLM token constraints.

- Unlike LLM-based relation extraction (slow, expensive, non-deterministic), this uses static compiler syntax rules
- Yields **>60× speedup** in graph construction while improving retrieval accuracy
- Produces identical graphs from identical codebases — fully auditable and reproducible

---

## Research & Publication

**Planned publication venues:**

| Venue | Type | Focus |
|---|---|---|
| **ICSE 2026/2027** — International Conference on Software Engineering | Conference paper | Special track on LLMs and Software Engineering |
| **EMSE** — Empirical Software Engineering | Journal article | Comparative benchmark: DKB-RAG vs LLM-KB vs Baseline RAG |

The quantitative benchmark data in `kg and others/json/` forms the empirical basis for both submissions.

---

## Team

**Samsung PRISM Work-let — Department of Computer Science and Engineering**

| Role | Name |
|---|---|
| **Faculty Mentor** | Prof. A K Verma |
| **Student Engineer** | Aditya Kumar (1024170314) |
| **Student Engineer** | Rehan Bansal |
| **Student Engineer** | Mehak Garg |
| **Student Engineer** | Manat Garg |

**Work-let Duration:** 6 months  
**Closure Date:** 24 June 2026

---

## Acknowledgements

This project was developed as part of the **Samsung PRISM (Prepare, Research, Innovate, and Scale with Mentors)** programme, in collaboration with Samsung Research Institute Bangalore (SRIB).

We acknowledge the open-source communities behind:
- [ThingsBoard](https://github.com/thingsboard/thingsboard) — IoT platform used as evaluation codebase
- [OpenMRS Core](https://github.com/openmrs/openmrs-core) — Medical records system used as evaluation codebase
- [Shopizer](https://github.com/shopizer-ecommerce/shopizer) — E-commerce platform used as evaluation codebase
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) — AST parsing framework
- [Neo4j](https://neo4j.com) — Graph database and vector index

---

<div align="center">

**Samsung PRISM · Code RAG using Knowledge Graph and Vector Database**  
Department of Computer Science and Engineering · June 2026

</div>
