# 🌙 SleepMind AI — Sleep-Time Compute Research Assistant

A production-quality implementation of the **Sleep-Time Compute** paradigm: instead of doing all the work when a user asks a question, SleepMind AI pre-processes research papers *offline*, before any query arrives, using four autonomous AI agents.

## What is Sleep-Time Compute?

| Paradigm | When it runs | What it does |
|---|---|---|
| Traditional RAG | At query time | Retrieve → Answer |
| Sleep-Time RAG | Before query time | Pre-analyze → Store → Retrieve (fast) → Answer |

By shifting analysis to "sleep time," the system answers user questions faster and cheaper, since most of the heavy reasoning already happened in advance.

## How it works

1. **PDF Ingestion Engine** — extracts, cleans, chunks, and embeds a research PDF (PyMuPDF + LangChain + FAISS).
2. **Sleep-Time Compute Engine (4 agents)** — runs autonomously over the ingested paper to produce summaries, FAQs, predicted questions, and extracted concepts.
3. **Knowledge Graph Builder** — turns extracted concepts into an interactive graph (NetworkX + PyVis).
4. **Sleep-Time Knowledge Store** — persists all generated artifacts (summaries, FAQs, predicted queries, graph data, chunk metadata) to disk.
5. **QA Pipelines + Benchmarking Engine** — compares traditional test-time RAG against the sleep-time pipeline on speed, token usage, and answer quality.
6. **Orchestrator (`SleepMindAI`)** — single entry point tying it all together.

## Quick start

```python
ai = SleepMindAI(CFG)
ai.ingest_pdf("/content/paper.pdf")
ai.run_sleep_time_compute()
ai.visualize_knowledge_graph()
ai.ask("What is sleep-time compute?", pipeline="both")
```

## Setup

```bash
pip install -q openai langchain langchain-openai langchain-community langchain-text-splitters langchain-core
pip install -q faiss-cpu pymupdf tiktoken
pip install -q networkx pyvis pandas numpy matplotlib
pip install -q python-dotenv tqdm colorama tabulate
```

**API key:** set your OpenAI key as an environment variable — never hardcode it in the notebook:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
import os
CFG.openai_api_key = os.getenv("OPENAI_API_KEY")
```

> ⚠️ If you're using Google Colab, use Colab Secrets instead of pasting your key directly into a cell.

## Benchmarking

The included benchmark compares the traditional and sleep-time pipelines across a question set (e.g. AIME, GSM-Symbolic-style questions), reporting latency, token usage, and cost. A full run of 10 questions × 2 pipelines costs roughly $0.01–0.05 depending on answer length.

## Interactive chat

```python
ai.interactive_chat(pipeline="sleep_time")
```

Commands: `switch` (toggle pipeline), `compare <question>` (run both pipelines), `exit` / `quit`.

## Requirements

- Python 3
- OpenAI API key
- Jupyter or Google Colab

## License

TBD
