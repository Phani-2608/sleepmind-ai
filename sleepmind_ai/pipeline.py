"""Top-level orchestrator: ties all components together.

Run with:  python -m sleepmind_ai.pipeline --pdf path/to/paper.pdf
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_openai import OpenAIEmbeddings

from .agents.engine import SleepTimeComputeEngine
from .config import DEFAULT_CONFIG, SleepMindConfig, resolve_api_key
from .evaluation.benchmark import BenchmarkingEngine
from .ingestion.pdf_engine import PDFIngestionEngine
from .knowledge_graph.builder import KnowledgeGraphBuilder
from .monitoring.run_log import RunLog
from .retrieval.qa_pipelines import SleepTimeQAPipeline, TraditionalRAGPipeline
from .storage.artifact_store import ArtifactStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class SleepMindAI:
    """Single entry point for the full SleepMind pipeline."""

    def __init__(self, cfg: SleepMindConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=resolve_api_key(cfg),
        )
        self.pdf_engine = PDFIngestionEngine(cfg)
        self.sleep_engine = SleepTimeComputeEngine(cfg)
        self.kg_builder = KnowledgeGraphBuilder(cfg)
        self.store = ArtifactStore(cfg.storage_dir)
        self.sleep_qa: SleepTimeQAPipeline | None = None
        self.traditional_qa: TraditionalRAGPipeline | None = None
        self.benchmark_engine: BenchmarkingEngine | None = None
        self._sleep_complete = False
        self._ingest_summary: dict | None = None
        self._sleep_artifacts: dict | None = None
        logger.info("SleepMind AI initialized (model=%s)", cfg.chat_model)

    def ingest_pdf(self, pdf_path: str) -> dict[str, Any]:
        logger.info("Ingesting: %s", os.path.basename(pdf_path))
        summary = self.pdf_engine.ingest(pdf_path)
        self.traditional_qa = TraditionalRAGPipeline(self.cfg, self.pdf_engine)
        # Persist FAISS index
        self.pdf_engine.save_vectorstore(self.cfg.faiss_dir)
        self._ingest_summary = summary
        logger.info(
            "Ingestion complete: %s chunks in %.2fs",
            summary["total_chunks"],
            summary["ingestion_time_sec"],
        )
        return summary

    def run_sleep_time_compute(self) -> dict[str, Any]:
        if not self.pdf_engine.raw_text:
            raise RuntimeError("No PDF ingested. Run ingest_pdf() first.")
        artifacts = self.sleep_engine.run_all(self.pdf_engine.raw_text)

        # Build knowledge graph
        self.kg_builder.build_from_agent_output(self.sleep_engine.knowledge_graph_data)

        # Persist all artifacts with content hashes
        chunk_meta = [c.metadata for c in self.pdf_engine.chunks]
        self.store.save_all(
            summary=self.sleep_engine.summary,
            faqs=self.sleep_engine.faqs,
            predicted_queries=self.sleep_engine.predicted_queries,
            knowledge_graph_data=self.sleep_engine.knowledge_graph_data,
            chunk_metadata=chunk_meta,
            session_meta={
                "total_sleep_time_sec": self.sleep_engine.total_elapsed,
                "faq_count": len(self.sleep_engine.faqs),
                "predicted_count": len(self.sleep_engine.predicted_queries),
                "run_version": self.cfg.run_version,
            },
        )

        # Build QA pipelines
        self.sleep_qa = SleepTimeQAPipeline(
            self.cfg,
            self.pdf_engine,
            self.store,
            self.sleep_engine,
            self.embeddings,
        )
        self.benchmark_engine = BenchmarkingEngine(self.cfg, self.sleep_qa, self.traditional_qa)
        self._sleep_complete = True
        self._sleep_artifacts = artifacts
        return artifacts

    def ask(self, question: str, pipeline: str = "sleep_time") -> dict[str, Any]:
        if pipeline == "sleep_time":
            if not self._sleep_complete:
                raise RuntimeError("Run run_sleep_time_compute() first.")
            return self.sleep_qa.answer(question).to_dict()
        elif pipeline == "traditional":
            if self.traditional_qa is None:
                raise RuntimeError("Run ingest_pdf() first.")
            return self.traditional_qa.answer(question).to_dict()
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}. Use 'sleep_time' or 'traditional'.")

    def compare(self, question: str) -> dict[str, Any]:
        trad = self.ask(question, "traditional")
        sleep = self.ask(question, "sleep_time")
        return {
            "question": question,
            "traditional": trad,
            "sleep_time": sleep,
            "delta_latency_sec": round(sleep["latency_sec"] - trad["latency_sec"], 3),
            "delta_tokens": sleep["total_tokens"] - trad["total_tokens"],
            "delta_cost": round(sleep["estimated_cost_usd"] - trad["estimated_cost_usd"], 6),
            "delta_sources": sleep["retrieved_chunks"].__len__()
            + sleep["retrieved_faqs"].__len__()
            + sleep["retrieved_predictions"].__len__()
            - len(trad["retrieved_chunks"]),
        }

    def benchmark(self, questions=None) -> dict[str, Any]:
        if self.benchmark_engine is None:
            raise RuntimeError("Run run_sleep_time_compute() first.")
        df = self.benchmark_engine.run(questions)
        summary = self.benchmark_engine.summary(df)

        # Cost breakeven
        preprocess_cost = sum(
            self.sleep_engine.summary.get("_usage", {}).get("total_tokens", 0) * 0.15 / 1e6
            for _ in range(4)  # rough estimate for all 4 agents
        )
        breakeven = self.benchmark_engine.cost_breakeven(preprocess_cost, df)

        # Log to MLflow
        self.benchmark_engine.log_to_mlflow(df, preprocess_cost)

        # Log to run history for regression detection
        run_log = RunLog(os.path.join(self.cfg.storage_dir, "run_log.jsonl"))
        run_log.append(summary, self.cfg.run_version)

        # Save benchmark CSV
        df.to_csv(os.path.join(self.cfg.storage_dir, "benchmark_results.csv"), index=False)

        return {
            "summary": summary,
            "breakeven": breakeven,
            "per_question": df.to_dict(orient="records"),
        }

    def get_knowledge_graph(self) -> dict[str, Any]:
        return self.kg_builder.to_serializable()

    def get_artifacts_status(self) -> list:
        return self.store.list_artifacts()


def run_pipeline(
    pdf_path: str, cfg: SleepMindConfig = DEFAULT_CONFIG, benchmark: bool = False
) -> dict[str, Any]:
    """Run the full pipeline from CLI."""
    ai = SleepMindAI(cfg)
    ingest = ai.ingest_pdf(pdf_path)
    artifacts = ai.run_sleep_time_compute()
    result = {
        "ingestion": ingest,
        "sleep_time_sec": artifacts["total_sleep_time_sec"],
        "faq_count": len(ai.sleep_engine.faqs),
        "predicted_count": len(ai.sleep_engine.predicted_queries),
        "kg_metrics": ai.kg_builder.compute_metrics(),
    }
    if benchmark:
        result["benchmark"] = ai.benchmark()
    # Save pipeline results
    os.makedirs(cfg.storage_dir, exist_ok=True)
    with open(os.path.join(cfg.storage_dir, "pipeline_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Pipeline complete. Results saved to %s/", cfg.storage_dir)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SleepMind AI Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to process")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run the benchmark after processing"
    )
    args = parser.parse_args()
    run_pipeline(args.pdf, benchmark=args.benchmark)
