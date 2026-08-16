"""Shared LLM call helpers with retry, timeout, and structured JSON extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output (handles markdown fences, preamble)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, cleaned)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


def call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 120.0,
) -> tuple[str, dict]:
    """Call the OpenAI chat API, returning (content_text, usage_dict).

    Raises TimeoutError if the call exceeds `timeout` seconds (measured client-side).
    """
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=timeout,
    )
    elapsed = time.time() - t0
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_sec": round(elapsed, 3),
    }
    return response.choices[0].message.content.strip(), usage
