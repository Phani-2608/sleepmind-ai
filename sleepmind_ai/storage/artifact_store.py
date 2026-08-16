"""Persistent structured store for all sleep-time artifacts and FAISS indexes."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

ARTIFACT_FILES = {
    "summary": "summary.json",
    "faqs": "faqs.json",
    "predicted_queries": "predicted_queries.json",
    "knowledge_graph_data": "knowledge_graph.json",
    "chunk_metadata": "chunk_metadata.json",
    "session": "session_metadata.json",
}


class ArtifactStore:
    """Filesystem-backed store with content hashing for reproducibility tracking."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> str:
        if key not in ARTIFACT_FILES:
            raise ValueError(f"Unknown artifact key: {key}")
        path = self.storage_dir / ARTIFACT_FILES[key]
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        path.write_text(payload, encoding="utf-8")
        content_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        self._store[key] = data
        logger.info("Saved %s -> %s (hash %s)", key, path, content_hash)
        return content_hash

    def save_all(
        self,
        summary: Any,
        faqs: Any,
        predicted_queries: Any,
        knowledge_graph_data: Any,
        chunk_metadata: Any,
        session_meta: dict | None = None,
    ) -> dict[str, str]:
        hashes = {}
        hashes["summary"] = self.save("summary", summary)
        hashes["faqs"] = self.save("faqs", faqs)
        hashes["predicted_queries"] = self.save("predicted_queries", predicted_queries)
        hashes["knowledge_graph_data"] = self.save("knowledge_graph_data", knowledge_graph_data)
        hashes["chunk_metadata"] = self.save("chunk_metadata", chunk_metadata)
        meta = session_meta or {}
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        meta["content_hashes"] = hashes
        self.save("session", meta)
        return hashes

    def load(self, key: str) -> Any | None:
        if key not in ARTIFACT_FILES:
            raise ValueError(f"Unknown artifact key: {key}")
        path = self.storage_dir / ARTIFACT_FILES[key]
        if not path.exists():
            logger.warning("Artifact not found: %s", path)
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        self._store[key] = data
        return data

    def load_all(self) -> dict[str, Any]:
        loaded = {}
        for key in ARTIFACT_FILES:
            loaded[key] = self.load(key)
        n = sum(1 for v in loaded.values() if v is not None)
        logger.info("Loaded %s/%s artifacts from %s", n, len(ARTIFACT_FILES), self.storage_dir)
        return loaded

    def get(self, key: str) -> Any:
        if self._store.get(key):
            return self._store[key]
        return self.load(key)

    def list_artifacts(self) -> list[dict[str, Any]]:
        rows = []
        for key, fname in ARTIFACT_FILES.items():
            path = self.storage_dir / fname
            if path.exists():
                rows.append(
                    {
                        "artifact": key,
                        "file": fname,
                        "size_kb": round(path.stat().st_size / 1024, 2),
                        "exists": True,
                    }
                )
            else:
                rows.append({"artifact": key, "file": fname, "size_kb": 0, "exists": False})
        return rows


def build_faq_index(faqs: list[dict], embeddings: OpenAIEmbeddings) -> LangFAISS | None:
    if not faqs:
        return None
    docs = [
        Document(
            page_content=f["question"],
            metadata={
                "answer": f.get("answer", ""),
                "category": f.get("category", ""),
                "difficulty": f.get("difficulty", ""),
                "faq_id": f.get("id", i),
            },
        )
        for i, f in enumerate(faqs)
    ]
    vs = LangFAISS.from_documents(docs, embeddings)
    logger.info("FAQ index built: %s entries", len(docs))
    return vs


def build_prediction_index(
    predictions: list[dict], embeddings: OpenAIEmbeddings
) -> LangFAISS | None:
    if not predictions:
        return None
    docs = [
        Document(
            page_content=p["question"],
            metadata={
                "predicted_answer": p.get("predicted_answer", ""),
                "rank": p.get("rank", 99),
                "likelihood_score": p.get("likelihood_score", 0.0),
            },
        )
        for p in predictions
    ]
    vs = LangFAISS.from_documents(docs, embeddings)
    logger.info("Prediction index built: %s entries", len(docs))
    return vs
