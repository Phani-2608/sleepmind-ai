"""Sleep-time compute engine: orchestrates all four agents."""

from __future__ import annotations

import logging
import time
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
    """Orchestrates all four sleep-time agents sequentially."""

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
        logger.info("Sleep-time compute starting...")
        t0 = time.time()

        logger.info("[1/4] SummaryAgent")
        self.summary = self.summary_agent.run(raw_text)

        logger.info("[2/4] FAQGeneratorAgent")
        self.faqs = self.faq_agent.run(raw_text)

        logger.info("[3/4] FutureQueryPredictorAgent")
        self.predicted_queries = self.query_predictor.run(raw_text)

        logger.info("[4/4] ConceptExtractorAgent")
        self.knowledge_graph_data = self.concept_extractor.run(raw_text)

        self.total_elapsed = round(time.time() - t0, 2)
        logger.info("Sleep-time compute complete in %.2fs", self.total_elapsed)

        return {
            "summary": self.summary,
            "faqs": self.faqs,
            "predicted_queries": self.predicted_queries,
            "knowledge_graph_data": self.knowledge_graph_data,
            "total_sleep_time_sec": self.total_elapsed,
        }
