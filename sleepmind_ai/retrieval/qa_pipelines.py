"""QA pipelines: Sleep-Time RAG and Traditional RAG."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from ..agents.engine import SleepTimeComputeEngine
from ..config import SleepMindConfig, resolve_api_key
from ..ingestion.pdf_engine import PDFIngestionEngine
from ..storage.artifact_store import ArtifactStore, build_faq_index, build_prediction_index

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    question: str
    answer: str
    pipeline: str
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieved_faqs: list[str] = field(default_factory=list)
    retrieved_predictions: list[str] = field(default_factory=list)
    context_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    estimated_cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sources_count(self) -> int:
        return (
            len(self.retrieved_chunks) + len(self.retrieved_faqs) + len(self.retrieved_predictions)
        )


def _estimate_cost(prompt_tokens: int, completion_tokens: int, cfg: SleepMindConfig) -> float:
    return round(
        prompt_tokens / 1_000_000 * cfg.cost_input_per_1m
        + completion_tokens / 1_000_000 * cfg.cost_output_per_1m,
        6,
    )


class SleepTimeQAPipeline:
    """Answers questions using sleep-time artifacts (summary, FAQs, predicted Q&A) plus source chunks."""

    SYSTEM_PROMPT = (
        "You are SleepMind AI, an expert research assistant. "
        "You have comprehensive pre-computed knowledge about a research paper including "
        "summaries, FAQs, and predicted Q&A pairs generated before this conversation "
        "(Sleep-Time Compute). Be precise, cite specific findings, and acknowledge uncertainty."
    )
    USER_TEMPLATE = """## QUESTION
{question}

## PRE-COMPUTED PAPER SUMMARY (Sleep-Time Artifact)
{summary}

## RELEVANT DOCUMENT CHUNKS (Retrieved)
{chunks}

## RELEVANT PRE-COMPUTED FAQs (Sleep-Time Artifact)
{faqs}

## RELEVANT PREDICTED Q&A PAIRS (Sleep-Time Artifact)
{predictions}

## INSTRUCTION
Using ALL context above, provide a comprehensive, accurate answer."""

    def __init__(
        self,
        cfg: SleepMindConfig,
        pdf_engine: PDFIngestionEngine,
        store: ArtifactStore,
        sleep_engine: SleepTimeComputeEngine,
        embeddings: OpenAIEmbeddings,
    ):
        self.cfg = cfg
        self.pdf_engine = pdf_engine
        self.store = store
        self.sleep_engine = sleep_engine
        self.client = OpenAI(api_key=resolve_api_key(cfg))
        self.faq_index: LangFAISS | None = None
        self.prediction_index: LangFAISS | None = None
        self._build_indexes(embeddings)

    def _build_indexes(self, embeddings: OpenAIEmbeddings) -> None:
        if self.sleep_engine.faqs:
            self.faq_index = build_faq_index(self.sleep_engine.faqs, embeddings)
        if self.sleep_engine.predicted_queries:
            self.prediction_index = build_prediction_index(
                self.sleep_engine.predicted_queries, embeddings
            )

    def _retrieve_chunks(self, query: str) -> list[str]:
        results = self.pdf_engine.similarity_search(query, k=self.cfg.top_k_chunks)
        return [doc.page_content for doc, _ in results]

    def _retrieve_faqs(self, query: str) -> list[str]:
        if self.faq_index is None:
            return []
        results = self.faq_index.similarity_search_with_score(query, k=self.cfg.top_k_faqs)
        return [f"Q: {doc.page_content}\nA: {doc.metadata.get('answer', '')}" for doc, _ in results]

    def _retrieve_predictions(self, query: str) -> list[str]:
        if self.prediction_index is None:
            return []
        results = self.prediction_index.similarity_search_with_score(
            query, k=self.cfg.top_k_predicted
        )
        return [
            f"Predicted Q: {doc.page_content}\nPre-computed A: {doc.metadata.get('predicted_answer', '')}"
            for doc, _ in results
        ]

    def _summary_context(self) -> str:
        s = self.sleep_engine.summary
        if not s:
            return "(No summary)"
        parts = [s.get("one_line_summary", ""), "\nKey Findings:"]
        for f in s.get("key_findings", [])[:4]:
            parts.append(f"- {f.get('finding', f) if isinstance(f, dict) else f}")
        return "\n".join(parts)

    def answer(self, question: str) -> QAResult:
        t0 = time.time()
        chunks = self._retrieve_chunks(question)
        faqs = self._retrieve_faqs(question)
        predictions = self._retrieve_predictions(question)
        user_msg = self.USER_TEMPLATE.format(
            question=question,
            summary=self._summary_context(),
            chunks="\n\n---\n\n".join([f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunks)])
            or "(No chunks)",
            faqs="\n\n".join(faqs) or "(No FAQs)",
            predictions="\n\n".join(predictions) or "(No predictions)",
        )
        response = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens_answer,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        usage = response.usage
        return QAResult(
            question=question,
            answer=response.choices[0].message.content.strip(),
            pipeline="sleep_time",
            retrieved_chunks=chunks,
            retrieved_faqs=faqs,
            retrieved_predictions=predictions,
            context_tokens=int(len(user_msg.split()) * 1.3),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_sec=round(time.time() - t0, 3),
            estimated_cost_usd=_estimate_cost(
                usage.prompt_tokens, usage.completion_tokens, self.cfg
            ),
        )


class TraditionalRAGPipeline:
    """Standard retrieve-then-generate pipeline with no sleep-time artifacts."""

    SYSTEM_PROMPT = (
        "You are a research paper QA assistant. "
        "Answer questions based only on the provided document excerpts. "
        "Be accurate, concise, and cite evidence from the context."
    )
    USER_TEMPLATE = """## QUESTION
{question}

## RETRIEVED DOCUMENT CHUNKS
{chunks}

## INSTRUCTION
Answer the question based on the retrieved chunks above."""

    def __init__(self, cfg: SleepMindConfig, pdf_engine: PDFIngestionEngine):
        self.cfg = cfg
        self.pdf_engine = pdf_engine
        self.client = OpenAI(api_key=resolve_api_key(cfg))

    def answer(self, question: str) -> QAResult:
        t0 = time.time()
        results = self.pdf_engine.similarity_search(question, k=self.cfg.top_k_chunks)
        chunks = [doc.page_content for doc, _ in results]
        chunks_text = "\n\n---\n\n".join([f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunks)])
        user_msg = self.USER_TEMPLATE.format(question=question, chunks=chunks_text)
        response = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens_answer,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        usage = response.usage
        return QAResult(
            question=question,
            answer=response.choices[0].message.content.strip(),
            pipeline="traditional",
            retrieved_chunks=chunks,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_sec=round(time.time() - t0, 3),
            estimated_cost_usd=_estimate_cost(
                usage.prompt_tokens, usage.completion_tokens, self.cfg
            ),
        )
