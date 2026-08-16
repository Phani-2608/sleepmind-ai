# SleepMind AI

**Agentic RAG with Sleep-Time Compute.** Instead of doing all the reasoning when a user asks a question, SleepMind AI pre-processes research papers *offline* using four autonomous agents, so queries are answered faster and cheaper, since most of the heavy thinking already happened in advance.

| Paradigm | When it runs | What it does |
|---|---|---|
| Traditional RAG | At query time | Retrieve → Answer |
| Sleep-Time RAG | Before query time | Pre-analyze → Store → Retrieve (fast) → Answer |

The system benchmarks both approaches head to head on the same question set, measuring latency, token usage, cost per query, retrieval sources, and a break-even model that shows when the upfront preprocessing pays for itself.

---

## What this project demonstrates

- **Agentic AI** — four autonomous agents (Summary, FAQ Generator, Query Predictor, Concept Extractor), each with retry logic, timeouts, structured JSON output, and graceful failure recovery
- **RAG architecture** — FAISS vector search over source chunks, pre-generated FAQ index, predicted-query index, and knowledge-graph context fused at query time
- **Knowledge graphs** — NetworkX graph with PageRank-based concept prioritization, serializable for API and dashboard consumption
- **LLM evaluation and experimentation** — controlled benchmark harness comparing Traditional RAG vs Sleep-Time RAG on latency, tokens, cost, and source coverage, with MLflow experiment tracking
- **Cost engineering** — break-even model that calculates when sleep-time preprocessing becomes cheaper than paying query-time cost repeatedly
- **FastAPI service** — endpoints for upload, preprocess, query, compare, artifact inspection, and knowledge-graph retrieval
- **Streamlit dashboard** — explore artifacts, knowledge graph, benchmark results, and ask questions interactively
- **MLOps** — MLflow experiment logging (local file-store, no server), run versioning, regression detection on latency and cost
- **Production engineering** — Docker, docker-compose, GitHub Actions CI (lint + tests + Docker build), structured logging, `.env`-based secrets management, content-hashed artifact storage
- **Testing** — 27 pytest tests covering config, all four agents, storage, knowledge graph, monitoring, and API, all mocked so CI runs without an API key

---

## Repository layout

```
sleepmind-ai/
├── sleepmind_ai/
│   ├── config.py                  # Central config + API key resolution
│   ├── pipeline.py                # Top-level orchestrator
│   ├── ingestion/
│   │   └── pdf_engine.py          # PDF extraction, chunking, FAISS indexing
│   ├── agents/
│   │   ├── base.py                # BaseAgent with retry/timeout/structured output
│   │   ├── sleep_agents.py        # SummaryAgent, FAQGenerator, QueryPredictor, ConceptExtractor
│   │   ├── engine.py              # Agent orchestrator
│   │   └── llm_helpers.py         # Shared LLM call + JSON extraction
│   ├── retrieval/
│   │   └── qa_pipelines.py        # SleepTimeQAPipeline + TraditionalRAGPipeline
│   ├── knowledge_graph/
│   │   └── builder.py             # NetworkX graph + PageRank + serialization
│   ├── storage/
│   │   └── artifact_store.py      # Persistent JSON store + FAISS index builders
│   ├── evaluation/
│   │   └── benchmark.py           # Controlled benchmark + cost breakeven + MLflow
│   ├── api/
│   │   └── service.py             # FastAPI service
│   ├── dashboard/
│   │   └── app.py                 # Streamlit dashboard
│   └── monitoring/
│       └── run_log.py             # Run history + regression detection
├── tests/                         # 27 pytest tests (all mocked, no API key)
├── configs/default.yaml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/Phani-2608/sleepmind-ai.git
cd sleepmind-ai
pip install -r requirements.txt
pip install -e .

# Set your API key (never hardcode it)
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-your-real-key
export $(cat .env | xargs)

# Run the full pipeline on a PDF
python -m sleepmind_ai.pipeline --pdf path/to/paper.pdf

# Run with benchmark (compares both RAG modes)
python -m sleepmind_ai.pipeline --pdf path/to/paper.pdf --benchmark
```

### API

```bash
uvicorn sleepmind_ai.api.service:app --port 8080

# Upload a PDF
curl -X POST http://localhost:8080/upload -F "file=@paper.pdf"

# Run sleep-time preprocessing
curl -X POST http://localhost:8080/preprocess

# Query (sleep-time or traditional)
curl -X POST http://localhost:8080/query \
  -H "content-type: application/json" \
  -d '{"question": "What is sleep-time compute?", "pipeline": "sleep_time"}'

# Side-by-side comparison
curl -X POST http://localhost:8080/compare \
  -H "content-type: application/json" \
  -d '{"question": "What are the main results?"}'
```

### Dashboard

```bash
pip install streamlit httpx
streamlit run sleepmind_ai/dashboard/app.py
```

### Docker

```bash
docker-compose up     # API on :8080, dashboard on :8501
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest                 # 27 tests, no API key needed
```

---

## How it works

**1. PDF Ingestion** — PyMuPDF extracts text, which is cleaned, chunked (800 char windows with 120 overlap), embedded (text-embedding-3-small), and indexed into FAISS.

**2. Sleep-Time Compute (4 agents)** — each agent runs independently over the raw paper text:
- **SummaryAgent** — structured summary with contributions, findings, limitations
- **FAQGeneratorAgent** — 20 Q&A pairs across difficulty levels and categories
- **FutureQueryPredictorAgent** — 15 predicted questions ranked by likelihood, with pre-computed answers
- **ConceptExtractorAgent** — concepts, entities, and relationships for a knowledge graph

All agents have retry logic, configurable timeouts, structured JSON output, and return a safe default on failure so the pipeline never crashes on a single agent.

**3. Knowledge Graph** — ConceptExtractor output is built into a NetworkX directed graph with PageRank scoring. Serializable for the API and dashboard.

**4. Artifact Persistence** — summaries, FAQs, predictions, and graph data are saved as JSON with content hashes for reproducibility. FAISS indexes are persisted to disk so the system warm-starts without re-embedding.

**5. Query Pipelines** — at query time, Traditional RAG retrieves only source chunks. Sleep-Time RAG fuses source chunks + pre-generated FAQs + predicted Q&A pairs + paper summary into a richer context window.

**6. Benchmark** — the same 10 questions run through both pipelines, measuring latency, token consumption, cost, and source count. Results are logged to MLflow (local file-store) for experiment tracking and to a JSONL run log for regression detection.

**7. Cost Break-Even Model** — calculates how many queries the system needs to serve before the upfront sleep-time compute cost is recovered through cheaper per-query inference. This is the key question for any system that shifts compute from query time to preprocessing: *when does it pay for itself?*

---

## Design principles

- **Every agent is testable independently.** All four agents inherit from `BaseAgent`, which provides retry, timeout, and structured output. Tests mock the LLM call, not the agent logic.
- **No hardcoded secrets.** API keys come from `.env` or environment variables, never from source code. `.env` is in `.gitignore`.
- **Train/serve consistency.** The same ingestion engine and FAISS index serve both the pipeline and the API.
- **CI runs without an API key.** All 27 tests are mocked. The CI workflow lints, tests, and builds Docker on every push.
- **Artifacts are content-hashed.** Every saved artifact gets a SHA-256 hash so you can trace which pipeline run and configuration produced it.

---

## License

MIT.
