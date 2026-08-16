"""Benchmarking engine: controlled comparison of Traditional RAG vs SleepMind RAG.

Measures latency, prompt/completion tokens, cost per query, retrieval source count,
and logs every run to MLflow (local file-store, no server required).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config import SleepMindConfig
from ..retrieval.qa_pipelines import SleepTimeQAPipeline, TraditionalRAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "What is sleep-time compute and how does it differ from test-time compute?",
    "What datasets were used to evaluate the approach?",
    "What are the main performance improvements reported in the paper?",
    "How does the system handle multiple queries about the same context?",
    "What are the key limitations of the sleep-time compute approach?",
    "How does sleep-time compute compare to pass@k parallel scaling?",
    "What models were evaluated in the experiments?",
    "How is query predictability defined and measured?",
    "What future research directions do the authors propose?",
    "What is the SWE-Features benchmark and how was it constructed?",
]


class BenchmarkingEngine:
    """Runs the same question set through both pipelines and records metrics."""

    def __init__(
        self,
        cfg: SleepMindConfig,
        sleep_pipeline: SleepTimeQAPipeline,
        traditional_pipeline: TraditionalRAGPipeline,
    ):
        self.cfg = cfg
        self.sleep_pipeline = sleep_pipeline
        self.traditional_pipeline = traditional_pipeline
        self.results: list[dict[str, Any]] = []

    def run(self, questions: list[str] | None = None) -> pd.DataFrame:
        questions = questions or DEFAULT_QUESTIONS
        self.results = []
        logger.info("Benchmarking: %s questions x 2 pipelines", len(questions))

        for i, q in enumerate(questions, 1):
            logger.info("[%s/%s] %s", i, len(questions), q[:60])
            trad = self.traditional_pipeline.answer(q)
            sleep = self.sleep_pipeline.answer(q)

            self.results.append(
                {
                    "question": q,
                    "trad_latency": trad.latency_sec,
                    "sleep_latency": sleep.latency_sec,
                    "trad_tokens": trad.total_tokens,
                    "sleep_tokens": sleep.total_tokens,
                    "trad_prompt_tokens": trad.prompt_tokens,
                    "sleep_prompt_tokens": sleep.prompt_tokens,
                    "trad_completion_tokens": trad.completion_tokens,
                    "sleep_completion_tokens": sleep.completion_tokens,
                    "trad_cost": trad.estimated_cost_usd,
                    "sleep_cost": sleep.estimated_cost_usd,
                    "trad_sources": trad.sources_count,
                    "sleep_sources": sleep.sources_count,
                    "trad_answer_len": len(trad.answer),
                    "sleep_answer_len": len(sleep.answer),
                }
            )

        df = pd.DataFrame(self.results)
        return df

    def summary(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Aggregate benchmark metrics."""
        df = df if df is not None else pd.DataFrame(self.results)
        if df.empty:
            return {}
        return {
            "n_questions": len(df),
            "avg_trad_latency": round(float(df["trad_latency"].mean()), 3),
            "avg_sleep_latency": round(float(df["sleep_latency"].mean()), 3),
            "latency_delta_pct": round(
                float(
                    (df["sleep_latency"].mean() - df["trad_latency"].mean())
                    / df["trad_latency"].mean()
                    * 100
                ),
                1,
            ),
            "avg_trad_tokens": round(float(df["trad_tokens"].mean()), 0),
            "avg_sleep_tokens": round(float(df["sleep_tokens"].mean()), 0),
            "total_trad_cost": round(float(df["trad_cost"].sum()), 5),
            "total_sleep_cost": round(float(df["sleep_cost"].sum()), 5),
            "avg_trad_sources": round(float(df["trad_sources"].mean()), 1),
            "avg_sleep_sources": round(float(df["sleep_sources"].mean()), 1),
        }

    def cost_breakeven(
        self, sleep_preprocessing_cost: float, df: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Calculate how many queries before sleep-time compute pays for itself.

        This is the key differentiator: showing when the upfront offline compute
        becomes cheaper than repeatedly paying higher query-time cost.
        """
        df = df if df is not None else pd.DataFrame(self.results)
        if df.empty:
            return {"breakeven_queries": float("inf")}

        avg_trad_cost = float(df["trad_cost"].mean())
        avg_sleep_cost = float(df["sleep_cost"].mean())
        cost_savings_per_query = avg_trad_cost - avg_sleep_cost

        if cost_savings_per_query <= 0:
            return {
                "breakeven_queries": float("inf"),
                "reason": "Sleep-time RAG costs more per query than traditional RAG.",
                "avg_trad_cost_per_query": avg_trad_cost,
                "avg_sleep_cost_per_query": avg_sleep_cost,
                "preprocessing_cost": sleep_preprocessing_cost,
            }

        breakeven = sleep_preprocessing_cost / cost_savings_per_query
        return {
            "breakeven_queries": round(breakeven, 0),
            "preprocessing_cost": sleep_preprocessing_cost,
            "avg_trad_cost_per_query": avg_trad_cost,
            "avg_sleep_cost_per_query": avg_sleep_cost,
            "savings_per_query": round(cost_savings_per_query, 6),
        }

    def log_to_mlflow(self, df: pd.DataFrame, sleep_preprocessing_cost: float = 0.0) -> None:
        """Log benchmark results to MLflow (local file-store, no server needed)."""
        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow not installed; skipping experiment tracking.")
            return

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)

        metrics = self.summary(df)
        breakeven = self.cost_breakeven(sleep_preprocessing_cost, df)

        with mlflow.start_run(run_name=f"benchmark-{self.cfg.run_version}"):
            mlflow.log_params(
                {
                    "model": self.cfg.chat_model,
                    "embedding_model": self.cfg.embedding_model,
                    "chunk_size": self.cfg.chunk_size,
                    "top_k_chunks": self.cfg.top_k_chunks,
                    "faq_count": self.cfg.faq_count,
                    "predicted_question_count": self.cfg.predicted_question_count,
                    "run_version": self.cfg.run_version,
                }
            )
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            if isinstance(breakeven.get("breakeven_queries"), (int, float)):
                mlflow.log_metric("breakeven_queries", breakeven["breakeven_queries"])
            df.to_csv("/tmp/benchmark_results.csv", index=False)
            mlflow.log_artifact("/tmp/benchmark_results.csv")
        logger.info("Benchmark logged to MLflow experiment '%s'", self.cfg.experiment_name)
