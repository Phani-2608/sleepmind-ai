"""Central configuration for SleepMind AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SleepMindConfig:
    # OpenAI
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    max_tokens_answer: int = 1500
    max_tokens_agent: int = 4096

    # PDF Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_length: int = 80

    # Retrieval
    top_k_chunks: int = 5
    top_k_faqs: int = 3
    top_k_predicted: int = 3

    # Sleep-Time Agents
    faq_count: int = 20
    predicted_question_count: int = 15
    agent_max_retries: int = 2
    agent_timeout_sec: float = 120.0

    # Storage
    storage_dir: str = "outputs/store"
    faiss_dir: str = "outputs/faiss"
    knowledge_graph_output: str = "outputs/knowledge_graph.html"

    # Cost (USD per 1M tokens, gpt-4o-mini defaults)
    cost_input_per_1m: float = 0.150
    cost_output_per_1m: float = 0.600

    # MLOps
    mlflow_tracking_uri: str = "outputs/mlruns"
    experiment_name: str = "sleepmind-benchmarks"
    run_version: str = "1.0.0"

    benchmark_questions: list[str] = field(default_factory=list)


def resolve_api_key(cfg: SleepMindConfig) -> str:
    """Resolve the OpenAI API key from config, env var, or Colab secrets — never hardcoded."""
    if cfg.openai_api_key:
        return cfg.openai_api_key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    try:
        from google.colab import userdata

        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    raise ValueError("API key not found. Set OPENAI_API_KEY in your environment or .env file.")


DEFAULT_CONFIG = SleepMindConfig()
