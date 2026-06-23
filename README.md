# 🧠 Unified Codebase RAG & GraphRAG Platform

Welcome to the unified **Codebase Retrieval-Augmented Generation (Codebase RAG)** repository. This workspace contains three distinct implementations and benchmarks comparing traditional semantic vector search against advanced, syntax-aware **Graph-based RAG (GraphRAG)** systems.

Traditional RAG struggles with source code because random text-splitting breaks syntax structures and loses relationship dependencies (e.g., function calls, inheritance, imports). The solutions in this repository address this problem by combining vector similarity indices with abstract syntax tree (AST) graphs.

---

## 📂 Project Subdirectories

| Subproject | Description | Interface | Stack |
| :--- | :--- | :--- | :--- |
| **[`hybrid rag`](./hybrid%20rag)** | **CodeGraph**: A desktop application for turning codebases into queryable Neo4j graphs with OpenAI embeddings. | Streamlit Dashboard | Streamlit, Neo4j, OpenAI (`gpt-4o`) |
| **[`kg and others`](./kg%20and%20others)** | **RAG Benchmarking Suite**: Quantitative evaluation scripts comparing DKB, LLM-KB, and baseline vector RAG. | CLI / JSON Outputs | Chroma DB, Gemini (`gemini-3-flash`) |
| **[`vector db code rag`](./vector%20db%20code%20rag)** | **CodeRAG AI**: A full-stack, session-isolated codebase RAG tool for multiple concurrent repositories. | React SPA Web UI | React, FastAPI, Neo4j, Redis, Gemini |

---

## 🏗️ Architecture Comparison
