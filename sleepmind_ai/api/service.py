"""FastAPI service for SleepMind AI.

Endpoints:
    GET  /health
    POST /upload          Upload + ingest a PDF
    POST /preprocess      Run sleep-time compute on the ingested PDF
    POST /query           Ask a question (sleep_time or traditional pipeline)
    POST /compare         Side-by-side comparison on a single question
    GET  /artifacts       List stored artifacts
    GET  /knowledge-graph Knowledge graph data (JSON)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import SleepMindConfig, resolve_api_key
from ..pipeline import SleepMindAI

logger = logging.getLogger("sleepmind_ai.api")

_STATE: dict[str, Any] = {}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    pipeline: str = Field("sleep_time", pattern="^(sleep_time|traditional)$")


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = SleepMindConfig()
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    cfg.storage_dir = os.environ.get("STORAGE_DIR", "outputs/store")
    cfg.faiss_dir = os.environ.get("FAISS_DIR", "outputs/faiss")
    try:
        resolve_api_key(cfg)
        _STATE["cfg"] = cfg
        _STATE["ai"] = None  # initialized on first upload
        logger.info("API ready (API key found)")
    except ValueError:
        _STATE["cfg"] = cfg
        _STATE["ai"] = None
        logger.warning("API started without an API key. Set OPENAI_API_KEY.")
    yield
    _STATE.clear()


app = FastAPI(
    title="SleepMind AI",
    version="1.0.0",
    description="Agentic RAG with Sleep-Time Compute: upload a PDF, preprocess with 4 agents, then query.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "internal_server_error"})


@app.get("/health")
def health():
    ai = _STATE.get("ai")
    return {
        "status": "ok",
        "pdf_ingested": ai is not None and ai._ingest_summary is not None,
        "sleep_complete": ai is not None and ai._sleep_complete,
    }


@app.post("/upload")
async def upload_pdf(file: Annotated[UploadFile, File()]):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    cfg = _STATE["cfg"]
    try:
        resolve_api_key(cfg)
    except ValueError:
        raise HTTPException(503, "No OpenAI API key configured.") from None

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ai = SleepMindAI(cfg)
        summary = ai.ingest_pdf(tmp_path)
        _STATE["ai"] = ai
        return {"status": "ingested", **summary}
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}") from e
    finally:
        os.unlink(tmp_path)


@app.post("/preprocess")
def preprocess():
    ai = _STATE.get("ai")
    if ai is None or ai._ingest_summary is None:
        raise HTTPException(400, "Upload a PDF first via /upload.")
    try:
        artifacts = ai.run_sleep_time_compute()
        return {
            "status": "complete",
            "sleep_time_sec": artifacts["total_sleep_time_sec"],
            "faq_count": len(ai.sleep_engine.faqs),
            "predicted_count": len(ai.sleep_engine.predicted_queries),
            "kg_nodes": ai.kg_builder.graph.number_of_nodes(),
            "kg_edges": ai.kg_builder.graph.number_of_edges(),
        }
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}") from e


@app.post("/query")
def query(req: QueryRequest):
    ai = _STATE.get("ai")
    if ai is None:
        raise HTTPException(400, "Upload and preprocess a PDF first.")
    try:
        return ai.ask(req.question, req.pipeline)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/compare")
def compare(req: CompareRequest):
    ai = _STATE.get("ai")
    if ai is None or not ai._sleep_complete:
        raise HTTPException(400, "Upload and preprocess a PDF first.")
    return ai.compare(req.question)


@app.get("/artifacts")
def artifacts():
    ai = _STATE.get("ai")
    if ai is None:
        return {"artifacts": []}
    return {"artifacts": ai.get_artifacts_status()}


@app.get("/knowledge-graph")
def knowledge_graph():
    ai = _STATE.get("ai")
    if ai is None or ai.kg_builder.graph.number_of_nodes() == 0:
        raise HTTPException(400, "Run /preprocess first to build the knowledge graph.")
    return ai.get_knowledge_graph()
