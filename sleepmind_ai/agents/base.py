"""Base agent with retry, timeout, structured output, and failure recovery."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from ..config import SleepMindConfig
from .llm_helpers import call_llm, extract_json

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """All four sleep-time agents inherit from this.

    Provides: retry loop, timeout enforcement, structured JSON output extraction,
    elapsed-time tracking, and a default failure payload so the pipeline never
    crashes on a single agent failure.
    """

    name: str = "BaseAgent"

    def __init__(self, cfg: SleepMindConfig, client: OpenAI):
        self.cfg = cfg
        self.client = client

    @abstractmethod
    def _system_prompt(self) -> str: ...

    @abstractmethod
    def _user_prompt(self, raw_text: str) -> str: ...

    @abstractmethod
    def _default_on_failure(self) -> Any: ...

    def _parse_output(self, raw_json: Any) -> Any:
        """Subclasses can override to normalize/validate the parsed JSON."""
        return raw_json

    def run(self, raw_text: str) -> Any:
        logger.info("[%s] Starting...", self.name)
        t0 = time.time()
        system = self._system_prompt()
        user = self._user_prompt(raw_text)

        for attempt in range(self.cfg.agent_max_retries + 1):
            try:
                content, usage = call_llm(
                    self.client,
                    system,
                    user,
                    model=self.cfg.chat_model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens_agent,
                    timeout=self.cfg.agent_timeout_sec,
                )
                result = self._parse_output(extract_json(content))
                elapsed = round(time.time() - t0, 2)
                logger.info("[%s] Done in %.2fs (attempt %s)", self.name, elapsed, attempt + 1)
                if isinstance(result, dict):
                    result["_agent"] = self.name
                    result["_elapsed_sec"] = elapsed
                    result["_generated_at"] = datetime.now(timezone.utc).isoformat()
                    result["_usage"] = usage
                return result

            except Exception as e:
                logger.warning("[%s] Attempt %s failed: %s", self.name, attempt + 1, e)
                if attempt == self.cfg.agent_max_retries:
                    logger.error("[%s] All attempts exhausted. Returning default.", self.name)
                    return self._default_on_failure()
        return self._default_on_failure()
