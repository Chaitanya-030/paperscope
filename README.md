# paperscope

**Agentic literature review + research gap finder.**

Give it a research topic. It plans the investigation, spawns sub-agents to
explore different angles, retrieves and caches papers in a vector database,
and produces a structured, cited report — including a taxonomy of prior work,
open problems, and candidate research directions.

Built to learn modern agentic-AI stacks end-to-end:
[LangChain](https://python.langchain.com/) ·
[deepagents](https://docs.langchain.com/oss/python/deepagents/overview) ·
[Qdrant](https://qdrant.tech/) ·
[LangSmith](https://smith.langchain.com/).

---

## What it produces

**Input:** a research topic, e.g. *"LLM-based agents for automated data quality monitoring"*

**Output:** a Markdown report containing
- **Landscape** — key papers, authors, venues, timeline
- **Taxonomy** — approaches grouped by method / application / dataset / evaluation
- **Gaps** — under-explored intersections and open problems
- **Directions** — 3–5 candidate research angles with justification and closest prior work
- **Citations** — every claim linked to a real arXiv / DOI source

---

## Why this project

Literature review is a universal research pain point and a validated product
space (Elicit, Consensus, SciSpace, Undermind.ai). It is also the canonical
use case the `deepagents` library was designed for, which makes it an honest
way to learn the full agentic stack — planning, sub-agents, skills,
retrieval, evaluation — without fighting the framework.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER: "topic X"                          │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  MAIN DEEPAGENT (planner / orchestrator)                         │
│    • write_todos  → breaks topic into sub-queries                │
│    • virtual FS   → scratchpad: sub_queries.md, clusters.md ...  │
│    • spawns sub-agents via `task` tool                           │
└──────────────────────────────────────────────────────────────────┘
           │              │                │               │
           ▼              ▼                ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  SEARCH    │  │  PAPER-    │  │  TAXONOMY  │  │  GAP       │
    │  SUB-AGENT │  │  ANALYST   │  │  BUILDER   │  │  ANALYZER  │
    │  (× per    │  │  (× per    │  │            │  │            │
    │  sub-query)│  │  paper)    │  │            │  │            │
    └────────────┘  └────────────┘  └────────────┘  └────────────┘
           │              │                │               │
           ▼              ▼                ▼               ▼
┌──────────────────────────────────────────────────────────────────┐
│  TOOLS                                                           │
│   arxiv_search · semantic_scholar · openalex · fetch_pdf ·       │
│   extract_sections · qdrant_upsert · qdrant_search               │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STATE                                                           │
│   Qdrant (papers + sections)  ·  virtual FS (per run)            │
│   LangSmith traces (per run)                                     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   REPORT WRITER      │
                    │   final_report.md    │
                    └──────────────────────┘
```

---

## 8-week build plan

| Week | Focus                                              | Deliverable                                                 |
|:----:|----------------------------------------------------|-------------------------------------------------------------|
| 1    | Python env + Claude API + first tool call         | Minimal script: Claude + arXiv search, with citations       |
| 2    | Embeddings + Qdrant + basic RAG                   | 30 abstracts ingested; retrieve-and-answer                  |
| 3    | LangChain + LangGraph + LangSmith tracing         | Multi-tool agent with cited answers                         |
| 4    | deepagents: planning + virtual filesystem         | Orchestrator that writes `todos.md` + scratchpad files      |
| 5    | Sub-agents (search / analyst / taxonomy / gap)    | End-to-end multi-agent run producing a sectioned report     |
| 6    | Skills — extract prompt-heavy logic into `SKILL.md` | Smaller base prompt, same-or-better output                 |
| 7    | Evaluation + cost tracking                        | Eval set of 10 topics, LLM-as-judge, $/run dashboard        |
| 8    | Streamlit UI + demo video + blog post             | Public portfolio artifact                                   |

---

## Design principles

- **LLM-portable.** Every model call goes through a thin abstraction so the
  planner/worker models are swappable via environment variables (Anthropic
  today, open models later) with no code changes.
- **Hallucination-hostile.** Every citation must round-trip through a tool
  call — the agent cannot invent references.
- **Bounded by design.** Hard caps on papers-per-run, sub-agent depth, and
  dollar budget. Agents fail closed, not open.
- **Traceable.** LangSmith tracing from Week 3 onward. Agents are effectively
  undebuggable without it.

---

## Status

Planning phase. Scaffolding, dependencies, and code come in subsequent commits.

## License

MIT
