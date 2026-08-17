"""Sleep-time compute engine: orchestrates all four agents."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAI

from ..config import SleepMindConfig, resolve_api_key
from .sleep_agents import (
    ConceptExtractorAgent,
    FAQGeneratorAgent,
    FutureQueryPredictorAgent,
    SummaryAgent,
)

logger = logging.getLogger(__name__)


class SleepTimeComputeEngine:
    """Orchestrates all four sleep-time agents.

    The four agents are independent — each only reads the raw document
    text and doesn't depend on any other agent's output — so they run
    concurrently instead of one after another. This turns total wait
    time from "sum of all four calls" into "roughly the slowest one".
    """

    def __init__(self, cfg: SleepMindConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=resolve_api_key(cfg))
        self.summary_agent = SummaryAgent(cfg, self.client)
        self.faq_agent = FAQGeneratorAgent(cfg, self.client)
        self.query_predictor = FutureQueryPredictorAgent(cfg, self.client)
        self.concept_extractor = ConceptExtractorAgent(cfg, self.client)

        self.summary: dict[str, Any] = {}
        self.faqs: list[dict[str, Any]] = []
        self.predicted_queries: list[dict[str, Any]] = []
        self.knowledge_graph_data: dict[str, Any] = {}
        self.total_elapsed: float = 0.0

    def run_all(self, raw_text: str) -> dict[str, Any]:
        logger.info("Sleep-time compute starting (4 agents running concurrently)...")
        t0 = time.time()

        jobs = {
            "summary": self.summary_agent,
            "faqs": self.faq_agent,
            "predicted_queries": self.query_predictor,
            "knowledge_graph_data": self.concept_extractor,
        }

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(agent.run, raw_text): name for name, agent in jobs.items()
            }
            for future in futures:
                name = futures[future]
                logger.info("Waiting on %s...", name)
            for future, name in futures.items():
                results[name] = future.result()
                logger.info("%s finished.", name)

        self.summary = results["summary"]
        self.faqs = results["faqs"]
        self.predicted_queries = results["predicted_queries"]
        self.knowledge_graph_data = results["knowledge_graph_data"]

        self.total_elapsed = round(time.time() - t0, 2)
        logger.info("Sleep-time compute complete in %.2fs", self.total_elapsed)

        return {
            "summary": self.summary,
            "faqs": self.faqs,
            "predicted_queries": self.predicted_queries,
            "knowledge_graph_data": self.knowledge_graph_data,
            "total_sleep_time_sec": self.total_elapsed,
        }
