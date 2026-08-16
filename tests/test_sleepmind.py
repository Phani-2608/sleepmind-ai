"""Tests for SleepMind AI — all LLM calls mocked so CI runs without an API key."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sleepmind_ai.agents.base import BaseAgent
from sleepmind_ai.agents.llm_helpers import extract_json
from sleepmind_ai.agents.sleep_agents import (
    ConceptExtractorAgent,
    FAQGeneratorAgent,
    FutureQueryPredictorAgent,
    SummaryAgent,
)
from sleepmind_ai.api.service import app
from sleepmind_ai.config import SleepMindConfig, resolve_api_key
from sleepmind_ai.knowledge_graph.builder import KnowledgeGraphBuilder
from sleepmind_ai.monitoring.run_log import RunLog
from sleepmind_ai.storage.artifact_store import ArtifactStore

# ──── Config ────────────────────────────────────────────────────


def test_resolve_api_key_from_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"}):
        cfg = SleepMindConfig()
        assert resolve_api_key(cfg) == "sk-test-123"


def test_resolve_api_key_from_config():
    cfg = SleepMindConfig(openai_api_key="sk-direct")
    assert resolve_api_key(cfg) == "sk-direct"


def test_resolve_api_key_raises_without_key():
    cfg = SleepMindConfig(openai_api_key="")

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key not found"):
            resolve_api_key(cfg)


# ──── LLM Helpers ───────────────────────────────────────────────


def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_markdown_fence():
    assert extract_json('```json\n{"b": 2}\n```') == {"b": 2}


def test_extract_json_with_preamble():
    assert extract_json('Here is the result:\n{"c": 3}\nDone.') == {"c": 3}


def test_extract_json_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        extract_json("this is not json at all")


# ──── Base Agent ────────────────────────────────────────────────


class _DummyAgent(BaseAgent):
    name = "DummyAgent"

    def _system_prompt(self):
        return "test"

    def _user_prompt(self, raw_text):
        return f"process: {raw_text[:50]}"

    def _default_on_failure(self):
        return {"error": True}


def test_agent_returns_default_on_all_retries_exhausted():
    cfg = SleepMindConfig(
        openai_api_key="sk-test",
        agent_max_retries=1,
    )

    client = MagicMock()
    agent = _DummyAgent(cfg, client)

    with patch(
        "sleepmind_ai.agents.base.call_llm",
        side_effect=Exception("boom"),
    ):
        result = agent.run("some text")

    assert result == {"error": True}


def test_agent_returns_parsed_json_on_success():
    cfg = SleepMindConfig(openai_api_key="sk-test")

    client = MagicMock()
    agent = _DummyAgent(cfg, client)

    mock_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "latency_sec": 0.5,
    }

    with patch(
        "sleepmind_ai.agents.base.call_llm",
        return_value=('{"result": "ok"}', mock_usage),
    ):
        result = agent.run("some text")

    assert result["result"] == "ok"
    assert result["_agent"] == "DummyAgent"
    assert "_elapsed_sec" in result


# ──── Sleep Agents ──────────────────────────────────────────────


@pytest.fixture
def mock_cfg():
    return SleepMindConfig(
        openai_api_key="sk-test",
        faq_count=5,
        predicted_question_count=3,
    )


def _mock_agent_run(agent_class, cfg, response_json):
    client = MagicMock()
    agent = agent_class(cfg, client)

    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "latency_sec": 0.3,
    }

    with patch(
        "sleepmind_ai.agents.base.call_llm",
        return_value=(json.dumps(response_json), usage),
    ):
        return agent.run("some paper text")


def test_summary_agent(mock_cfg):
    result = _mock_agent_run(
        SummaryAgent,
        mock_cfg,
        {
            "one_line_summary": "A great paper.",
            "executive_summary": "Details here.",
            "main_contributions": ["C1"],
            "key_findings": [
                {
                    "finding": "F1",
                    "significance": "S1",
                }
            ],
            "methodology": "M",
            "limitations": ["L1"],
            "future_work": ["FW1"],
            "paper_type": "empirical",
            "domain": "AI",
            "target_audience": "researchers",
        },
    )

    assert result["one_line_summary"] == "A great paper."


def test_faq_agent(mock_cfg):
    result = _mock_agent_run(
        FAQGeneratorAgent,
        mock_cfg,
        [
            {
                "id": 1,
                "question": "Q1?",
                "answer": "A1.",
                "category": "concept",
                "difficulty": "beginner",
                "keywords": ["k1"],
            }
        ],
    )

    assert isinstance(result, list)
    assert result[0]["question"] == "Q1?"


def test_query_predictor_agent(mock_cfg):
    result = _mock_agent_run(
        FutureQueryPredictorAgent,
        mock_cfg,
        [
            {
                "rank": 1,
                "question": "PQ?",
                "predicted_answer": "PA.",
                "audience": "researcher",
                "likelihood_score": 0.9,
                "rationale": "R",
            }
        ],
    )

    assert isinstance(result, list)
    assert result[0]["rank"] == 1


def test_concept_extractor_agent(mock_cfg):
    result = _mock_agent_run(
        ConceptExtractorAgent,
        mock_cfg,
        {
            "concepts": [
                {
                    "id": "C001",
                    "name": "STC",
                    "type": "method",
                    "description": "D",
                    "importance": "high",
                    "introduced_by_paper": True,
                }
            ],
            "entities": [
                {
                    "id": "E001",
                    "name": "GPT-4",
                    "type": "model",
                    "description": "D",
                }
            ],
            "relationships": [
                {
                    "source_id": "C001",
                    "target_id": "E001",
                    "relation": "uses",
                    "description": "D",
                    "weight": 1.0,
                }
            ],
        },
    )

    assert len(result["concepts"]) == 1


# ──── Storage ───────────────────────────────────────────────────


def test_artifact_store_save_load(tmp_path):
    store = ArtifactStore(str(tmp_path / "store"))

    content_hash = store.save(
        "summary",
        {"one_line": "test"},
    )

    assert len(content_hash) == 12

    loaded = store.load("summary")

    assert loaded["one_line"] == "test"


def test_artifact_store_list(tmp_path):
    store = ArtifactStore(str(tmp_path / "store"))

    store.save(
        "summary",
        {"x": 1},
    )

    listing = store.list_artifacts()

    found = [artifact for artifact in listing if artifact["artifact"] == "summary"]

    assert found[0]["exists"]


def test_artifact_store_unknown_key_raises(tmp_path):
    store = ArtifactStore(str(tmp_path / "store"))

    with pytest.raises(ValueError):
        store.save(
            "invalid_key",
            {},
        )


# ──── Knowledge Graph ───────────────────────────────────────────


def test_kg_builder_builds_graph():
    cfg = SleepMindConfig()
    builder = KnowledgeGraphBuilder(cfg)

    kg_data = {
        "concepts": [
            {
                "id": "C1",
                "name": "Sleep-Time",
                "type": "method",
                "description": "D",
                "importance": "high",
            },
            {
                "id": "C2",
                "name": "RAG",
                "type": "method",
                "description": "D",
                "importance": "medium",
            },
        ],
        "entities": [
            {
                "id": "E1",
                "name": "GPT-4",
                "type": "model",
                "description": "D",
            }
        ],
        "relationships": [
            {
                "source_id": "C1",
                "target_id": "C2",
                "relation": "extends",
                "weight": 1.0,
            },
            {
                "source_id": "C1",
                "target_id": "E1",
                "relation": "uses",
                "weight": 0.8,
            },
        ],
    }

    graph = builder.build_from_agent_output(kg_data)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_kg_metrics():
    cfg = SleepMindConfig()
    builder = KnowledgeGraphBuilder(cfg)

    builder.build_from_agent_output(
        {
            "concepts": [
                {
                    "id": "A",
                    "name": "A",
                },
                {
                    "id": "B",
                    "name": "B",
                },
            ],
            "entities": [],
            "relationships": [
                {
                    "source_id": "A",
                    "target_id": "B",
                    "relation": "r",
                }
            ],
        }
    )

    metrics = builder.compute_metrics()

    assert metrics["num_nodes"] == 2
    assert metrics["num_edges"] == 1
    assert "top_pagerank_nodes" in metrics


def test_kg_to_serializable():
    cfg = SleepMindConfig()
    builder = KnowledgeGraphBuilder(cfg)

    builder.build_from_agent_output(
        {
            "concepts": [
                {
                    "id": "X",
                    "name": "X",
                }
            ],
            "entities": [],
            "relationships": [],
        }
    )

    data = builder.to_serializable()

    assert "nodes" in data
    assert "edges" in data
    assert "metrics" in data


# ──── Monitoring ────────────────────────────────────────────────


def test_run_log_append_and_load(tmp_path):
    log = RunLog(str(tmp_path / "log.jsonl"))

    log.append(
        {"avg_sleep_latency": 1.2},
        "v1",
    )

    log.append(
        {"avg_sleep_latency": 1.3},
        "v2",
    )

    history = log.load_history()

    assert len(history) == 2
    assert history[0]["run_version"] == "v1"


def test_run_log_regression_detection(tmp_path):
    log = RunLog(str(tmp_path / "log.jsonl"))

    for _ in range(5):
        log.append(
            {
                "avg_sleep_latency": 1.0,
                "total_sleep_cost": 0.001,
            },
            "v1",
        )

    result = log.check_regression(
        {
            "avg_sleep_latency": 3.0,
            "total_sleep_cost": 0.001,
        },
        baseline_n=3,
        latency_threshold=1.5,
    )

    assert result["checked"]
    assert result["regressions_found"] > 0


def test_run_log_no_regression_on_normal_run(tmp_path):
    log = RunLog(str(tmp_path / "log.jsonl"))

    for _ in range(5):
        log.append(
            {
                "avg_sleep_latency": 1.0,
                "total_sleep_cost": 0.001,
            },
            "v1",
        )

    result = log.check_regression(
        {
            "avg_sleep_latency": 1.1,
            "total_sleep_cost": 0.001,
        },
        baseline_n=3,
    )

    assert result["regressions_found"] == 0


# ──── API (mocked, no OpenAI calls) ────────────────────────────


@pytest.fixture
def api_client():
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test-mock"},
    ):
        with TestClient(app) as client:
            yield client


def test_api_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_upload_rejects_non_pdf(api_client):
    response = api_client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_api_query_before_upload(api_client):
    response = api_client.post(
        "/query",
        json={
            "question": "What is X?",
            "pipeline": "traditional",
        },
    )

    assert response.status_code == 400


def test_api_artifacts_empty(api_client):
    response = api_client.get("/artifacts")

    assert response.status_code == 200
