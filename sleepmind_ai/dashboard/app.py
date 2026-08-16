"""Streamlit dashboard for SleepMind AI.

Run with:  streamlit run sleepmind_ai/dashboard/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SleepMind AI", layout="wide", page_icon="🌙")


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    st.title("🌙 SleepMind AI: Sleep-Time Compute Dashboard")

    store_dir = os.environ.get("STORAGE_DIR", "outputs/store")

    summary = _load_json(os.path.join(store_dir, "summary.json"))
    faqs = _load_json(os.path.join(store_dir, "faqs.json"))
    predicted = _load_json(os.path.join(store_dir, "predicted_queries.json"))
    kg_data = _load_json(os.path.join(store_dir, "knowledge_graph.json"))
    session = _load_json(os.path.join(store_dir, "session_metadata.json"))
    benchmark_path = os.path.join(store_dir, "benchmark_results.csv")

    if summary is None:
        st.warning(
            "No artifacts found. Run the pipeline first: `python -m sleepmind_ai.pipeline --pdf paper.pdf`"
        )
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FAQs generated", len(faqs) if faqs else 0)
    c2.metric("Predicted queries", len(predicted) if predicted else 0)
    sleep_sec = session.get("total_sleep_time_sec", "?") if session else "?"
    c3.metric("Sleep-time compute", f"{sleep_sec}s")
    kg_nodes = len(kg_data.get("concepts", [])) + len(kg_data.get("entities", [])) if kg_data else 0
    c4.metric("KG nodes", kg_nodes)

    tabs = st.tabs(["Summary", "FAQs", "Predicted Queries", "Knowledge Graph", "Benchmark"])

    with tabs[0]:
        st.subheader("Paper Summary")
        st.markdown(f"**{summary.get('one_line_summary', '')}**")
        st.write(summary.get("executive_summary", ""))
        st.subheader("Main Contributions")
        for c in summary.get("main_contributions", []):
            st.write(f"- {c}")
        st.subheader("Limitations")
        for lim in summary.get("limitations", []):
            st.write(f"- {lim}")

    with tabs[1]:
        st.subheader("Pre-generated FAQs (Sleep-Time Artifacts)")
        if faqs:
            for faq in faqs:
                with st.expander(f"Q{faq.get('id', '?')}: {faq.get('question', '')}"):
                    st.write(faq.get("answer", ""))
                    st.caption(
                        f"Category: {faq.get('category', '?')} | Difficulty: {faq.get('difficulty', '?')}"
                    )

    with tabs[2]:
        st.subheader("Predicted Future Queries (Sleep-Time Artifacts)")
        if predicted:
            for p in predicted:
                with st.expander(
                    f"#{p.get('rank', '?')}: {p.get('question', '')} (likelihood {p.get('likelihood_score', '?')})"
                ):
                    st.write(p.get("predicted_answer", ""))
                    st.caption(f"Audience: {p.get('audience', '?')} | {p.get('rationale', '')}")

    with tabs[3]:
        st.subheader("Knowledge Graph")
        if kg_data:
            concepts = kg_data.get("concepts", [])
            entities = kg_data.get("entities", [])
            rels = kg_data.get("relationships", [])
            st.write(
                f"{len(concepts)} concepts, {len(entities)} entities, {len(rels)} relationships"
            )
            if concepts:
                st.dataframe(
                    pd.DataFrame(concepts)[["id", "name", "type", "importance"]].head(20),
                    use_container_width=True,
                )
        else:
            st.info("No knowledge graph data available.")

    with tabs[4]:
        st.subheader("Benchmark: Traditional RAG vs Sleep-Time RAG")
        if os.path.exists(benchmark_path):
            df = pd.read_csv(benchmark_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Traditional latency", f"{df['trad_latency'].mean():.2f}s")
            c2.metric("Avg Sleep-Time latency", f"{df['sleep_latency'].mean():.2f}s")
            c3.metric(
                "Avg extra sources (Sleep)",
                f"+{(df['sleep_sources'] - df['trad_sources']).mean():.1f}",
            )
            st.bar_chart(df[["trad_latency", "sleep_latency"]])
            st.bar_chart(df[["trad_tokens", "sleep_tokens"]])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No benchmark results yet. Run with `--benchmark` flag.")

    # Interactive QA (requires running API)
    st.divider()
    st.subheader("Ask a question")
    st.caption("Requires the API to be running: `uvicorn sleepmind_ai.api.service:app --port 8080`")
    question = st.text_input("Your question:")
    pipeline = st.radio("Pipeline", ["sleep_time", "traditional", "compare"], horizontal=True)
    if st.button("Ask") and question:
        import httpx

        try:
            base = os.environ.get("API_URL", "http://localhost:8080")
            if pipeline == "compare":
                r = httpx.post(f"{base}/compare", json={"question": question}, timeout=60)
            else:
                r = httpx.post(
                    f"{base}/query", json={"question": question, "pipeline": pipeline}, timeout=60
                )
            if r.status_code == 200:
                st.json(r.json())
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Could not reach the API at {base}: {e}")


if __name__ == "__main__":
    main()
