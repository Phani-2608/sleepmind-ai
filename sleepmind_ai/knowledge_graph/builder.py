"""Knowledge graph builder: NetworkX + PageRank-based concept prioritization."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from ..config import SleepMindConfig

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and analyzes a directed knowledge graph from ConceptExtractorAgent output."""

    NODE_COLORS = {
        "method": "#4E9AF1",
        "dataset": "#F4A261",
        "metric": "#2A9D8F",
        "theory": "#9B59B6",
        "component": "#57C5B6",
        "result": "#E63946",
        "model": "#FF6B6B",
        "institution": "#95E1D3",
        "author": "#F8E71C",
        "benchmark": "#FD7F20",
        "framework": "#A8DADC",
        "default": "#B0BEC5",
    }

    def __init__(self, cfg: SleepMindConfig):
        self.cfg = cfg
        self.graph = nx.DiGraph()
        self.node_registry: dict[str, dict] = {}

    def build_from_agent_output(self, kg_data: dict[str, Any]) -> nx.DiGraph:
        self.graph.clear()
        self.node_registry.clear()

        for concept in kg_data.get("concepts", []):
            nid = concept.get("id", concept.get("name", ""))
            if not nid:
                continue
            self.graph.add_node(
                nid,
                label=concept["name"],
                node_type=concept.get("type", "default"),
                description=concept.get("description", ""),
                importance=concept.get("importance", "medium"),
                introduced=concept.get("introduced_by_paper", False),
                category="concept",
            )
            self.node_registry[nid] = concept

        for entity in kg_data.get("entities", []):
            nid = entity.get("id", entity.get("name", ""))
            if not nid:
                continue
            self.graph.add_node(
                nid,
                label=entity["name"],
                node_type=entity.get("type", "default"),
                description=entity.get("description", ""),
                importance="medium",
                introduced=False,
                category="entity",
            )
            self.node_registry[nid] = entity

        valid_nodes = set(self.graph.nodes())
        skipped = 0
        for rel in kg_data.get("relationships", []):
            src, tgt = rel.get("source_id", ""), rel.get("target_id", "")
            if src not in valid_nodes or tgt not in valid_nodes:
                skipped += 1
                continue
            self.graph.add_edge(
                src,
                tgt,
                relation=rel.get("relation", "related"),
                description=rel.get("description", ""),
                weight=rel.get("weight", 1.0),
            )
        logger.info(
            "KG: %s nodes, %s edges (%s skipped)",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            skipped,
        )
        return self.graph

    def compute_metrics(self) -> dict[str, Any]:
        if self.graph.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0}
        degree_dict = dict(self.graph.degree())
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except Exception:
            pagerank = {n: 0.0 for n in self.graph.nodes()}
        top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:5]
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": round(float(np.mean(list(degree_dict.values()))), 2),
            "density": round(nx.density(self.graph), 4),
            "top_pagerank_nodes": [
                {
                    "node": n,
                    "label": self.graph.nodes[n].get("label", n),
                    "score": round(pagerank[n], 4),
                }
                for n in top_nodes
            ],
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "num_weakly_connected": nx.number_weakly_connected_components(self.graph),
        }

    def to_serializable(self) -> dict[str, Any]:
        """Export the graph as a JSON-serializable dict (for API/dashboard)."""
        nodes = []
        for nid, attrs in self.graph.nodes(data=True):
            nodes.append({"id": nid, **{k: v for k, v in attrs.items()}})
        edges = []
        for src, tgt, attrs in self.graph.edges(data=True):
            edges.append({"source": src, "target": tgt, **{k: v for k, v in attrs.items()}})
        return {"nodes": nodes, "edges": edges, "metrics": self.compute_metrics()}
