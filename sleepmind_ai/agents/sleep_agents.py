"""The four sleep-time compute agents.

Each is a self-contained, testable component that inherits retry/timeout/structured
output from BaseAgent. Prompts are kept as class constants so they can be versioned.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import BaseAgent


class SummaryAgent(BaseAgent):
    name = "SummaryAgent"

    def _system_prompt(self) -> str:
        return (
            "You are a senior AI research scientist specializing in summarizing "
            "ML papers for expert audiences. "
            "You ALWAYS respond with valid, minified JSON only."
        )

    def _user_prompt(self, raw_text: str) -> str:
        return f"""Analyze the following research paper and produce a structured summary.

PAPER TEXT (first 12000 chars):
{raw_text[:12000]}

Respond with a single valid JSON object with exactly these keys:
{{
  "one_line_summary": "One sentence capturing the paper.",
  "executive_summary": "3-4 paragraph summary covering purpose, approach, results.",
  "main_contributions": ["contribution 1", "contribution 2"],
  "key_findings": [{{"finding": "...", "significance": "..."}}],
  "methodology": "Paragraph describing the experimental methodology.",
  "limitations": ["limitation 1", "limitation 2"],
  "future_work": ["direction 1", "direction 2"],
  "paper_type": "empirical|theoretical|survey|system",
  "domain": "primary research domain",
  "target_audience": "Who should read this"
}}"""

    def _default_on_failure(self) -> dict[str, Any]:
        return {"error": "SummaryAgent failed", "one_line_summary": "", "main_contributions": []}


class FAQGeneratorAgent(BaseAgent):
    name = "FAQGeneratorAgent"

    def _system_prompt(self) -> str:
        return (
            "You are a research Q&A expert. Generate clear, technically accurate "
            "question-answer pairs from academic papers. Respond with valid JSON only."
        )

    def _user_prompt(self, raw_text: str) -> str:
        return f"""Analyze the following research paper.

PAPER TEXT (first 10000 chars):
{raw_text[:10000]}

Generate exactly {self.cfg.faq_count} diverse, high-quality Q&A pairs covering:
concepts, methodology, results, comparisons, applications, and limitations.

Respond with a JSON array where each element has:
{{
  "id": 1,
  "question": "...",
  "answer": "Detailed answer (3-5 sentences).",
  "category": "concept|methodology|results|comparison|application|limitation",
  "difficulty": "beginner|intermediate|advanced",
  "keywords": ["kw1", "kw2"]
}}"""

    def _parse_output(self, raw_json: Any) -> list[dict]:
        if isinstance(raw_json, list):
            return raw_json
        if isinstance(raw_json, dict):
            return raw_json.get("faqs", raw_json.get("questions", []))
        return []

    def _default_on_failure(self) -> list:
        return []

    def run(self, raw_text: str) -> list[dict]:
        result = super().run(raw_text)
        if isinstance(result, list):
            for faq in result:
                faq["_generated_at"] = datetime.now(timezone.utc).isoformat()
        return result


class FutureQueryPredictorAgent(BaseAgent):
    name = "FutureQueryPredictorAgent"

    def _system_prompt(self) -> str:
        return (
            "You are an expert at predicting what questions researchers and "
            "practitioners ask after reading a paper. Think like a NeurIPS reviewer. "
            "Respond with valid JSON only."
        )

    def _user_prompt(self, raw_text: str) -> str:
        return f"""Analyze the following research paper.

PAPER TEXT (first 8000 chars):
{raw_text[:8000]}

Predict the {self.cfg.predicted_question_count} most likely questions a reader will ask, ranked 1=most likely.
Pre-compute high-quality answers for each.

Respond with a JSON array:
[
  {{
    "rank": 1,
    "question": "...",
    "predicted_answer": "Thorough answer (4-6 sentences).",
    "audience": "researcher|engineer|student|practitioner",
    "likelihood_score": 0.95,
    "rationale": "Why this question is likely."
  }}
]"""

    def _parse_output(self, raw_json: Any) -> list[dict]:
        if isinstance(raw_json, list):
            return raw_json
        if isinstance(raw_json, dict):
            return raw_json.get("predictions", [])
        return []

    def _default_on_failure(self) -> list:
        return []

    def run(self, raw_text: str) -> list[dict]:
        result = super().run(raw_text)
        if isinstance(result, list):
            for p in result:
                p["_generated_at"] = datetime.now(timezone.utc).isoformat()
        return result


class ConceptExtractorAgent(BaseAgent):
    name = "ConceptExtractorAgent"

    def _system_prompt(self) -> str:
        return (
            "You are a knowledge graph expert specializing in AI research. "
            "Extract structured concept-relationship triples from academic text. "
            "Respond with valid JSON only."
        )

    def _user_prompt(self, raw_text: str) -> str:
        return f"""Analyze this research paper and extract a rich knowledge graph.

PAPER TEXT (first 10000 chars):
{raw_text[:10000]}

Extract concepts, entities, and directed relationships.

Respond with:
{{
  "concepts": [
    {{
      "id": "C001",
      "name": "...",
      "type": "method|dataset|metric|theory|component|result",
      "description": "One-sentence definition.",
      "importance": "high|medium|low",
      "introduced_by_paper": true
    }}
  ],
  "entities": [
    {{
      "id": "E001",
      "name": "...",
      "type": "model|dataset|institution|author|benchmark|framework",
      "description": "..."
    }}
  ],
  "relationships": [
    {{
      "source_id": "C001",
      "target_id": "E001",
      "relation": "uses|extends|improves|evaluates|introduces|compares|requires",
      "description": "...",
      "weight": 1.0
    }}
  ]
}}"""

    def _default_on_failure(self) -> dict[str, Any]:
        return {"concepts": [], "entities": [], "relationships": []}
