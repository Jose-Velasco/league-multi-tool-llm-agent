# League Multi-Tool LLM Agent

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PydanticAI](https://img.shields.io/badge/PydanticAI-Graph%20Workflow-7c3aed)
![uv](https://img.shields.io/badge/package%20manager-uv-4c8bf5)
![Docker](https://img.shields.io/badge/devcontainer-Docker-2496ED)
![Postgres](https://img.shields.io/badge/database-PostgreSQL-336791)
![pgvector](https://img.shields.io/badge/vector-pgvector-0ea5e9)
![Status](https://img.shields.io/badge/status-early%20development-orange)

A graph-based, tool-augmented LLM assistant for **League of Legends** that aims to support player analysis, champion recommendations, builds, counters, and coaching workflows.

---

## Current Focus

This repository is currently focused on setting up the project foundation:

- Pydantic AI graph-based orchestration
- PostgreSQL / pgvector-backed retrieval
- Ollama-powered local model workflow
- Devcontainer-based reproducible development environment
- Gradio-ready interface path

---

## Architecture Preview

```mermaid
flowchart LR
    U[User Query] --> P[Parse / Route]
    P --> T[Tool Nodes]
    T --> A[Aggregate Context]
    A --> L[LLM Synthesis]
    L --> R[Response]
```

## Dev Environment

This project currently includes:

- a Docker/devcontainer workflow
- uv for dependency management
- local Ollama initialization for model serving
- Python 3.12 project configuration


## Quick Start
```bash
uv sync
```

If using the devcontainer, the environment setup is handled automatically during container creation.

## Planned Capabilities

- Player profile and match-history analysis
- Champion recommendation workflows
- Build and counter retrieval
- Patch/meta-aware responses
- Retrieval-augmented generation with structured + semantic sources