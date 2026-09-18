"""SleepMind AI - simple, non-technical web app.

Three plain steps: add a document, wait a moment while it's read,
then ask questions about it in your own words.

The API runs on a free hosting tier that sleeps after 15 minutes of
inactivity, so every call goes through ApiClient, which wakes the
backend with a cheap health check and retries transient edge failures
rather than giving up on the first one. See api_client.py.

Run with:  streamlit run sleepmind_ai/dashboard/app.py
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

from .api_client import ApiClient, ApiUnavailable, friendly_error

st.set_page_config(page_title="SleepMind AI", layout="centered", page_icon="🌙")

API_URL = os.environ.get("API_URL", "http://localhost:8080")
WAKE_TIMEOUT_SEC = int(os.environ.get("WAKE_TIMEOUT_SEC", "150"))

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main > div { max-width: 760px; margin: 0 auto; }
        h1 { font-size: 2.1rem !important; }
        .step-badge {
            display: inline-block; background: #eef2ff; color: #4338ca;
            border-radius: 999px; padding: 2px 12px; font-size: 0.85rem;
            font-weight: 600; margin-bottom: 6px;
        }
        .answer-box {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 20px 22px; font-size: 1.05rem; line-height: 1.55;
        }
        .subtle { color: #64748b; font-size: 0.9rem; }
        div.stButton > button {
            border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
defaults = {
    "stage": "upload",  # upload -> ready
    "doc_summary": None,
    "history": [],  # list of (question, answer_dict, error)
    "error": None,
    "api_awake": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_all():
    keep_awake = st.session_state.get("api_awake", False)
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state.api_awake = keep_awake


def get_client() -> ApiClient:
    """A client per rerun, carrying the awake flag across reruns."""
    client = ApiClient(API_URL, wake_timeout_sec=WAKE_TIMEOUT_SEC)
    client.awake = bool(st.session_state.get("api_awake", False))
    return client


def remember(client: ApiClient) -> None:
    st.session_state.api_awake = client.awake


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌙 SleepMind AI")
st.write(
    "Upload a document, and ask questions about it in plain English. "
    "SleepMind reads the whole thing in advance, so its answers are fast "
    "and grounded in what's actually written."
)
st.divider()

# ---------------------------------------------------------------------------
# Stage 1 — Upload
# ---------------------------------------------------------------------------
if st.session_state.stage == "upload":
    st.markdown('<span class="step-badge">Step 1 of 2</span>', unsafe_allow_html=True)
    st.subheader("Add your document")
    st.caption("Works best with a written PDF up to a few dozen pages — a report, paper, or manual.")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None:
        st.session_state.error = None
        client = get_client()
        with st.status("Reading your document...", expanded=True) as status:

            def note(message: str) -> None:
                status.update(label=message, state="running")

            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                r = client.post("/upload", progress=note, files=files)

                if r.status_code != 200:
                    status.update(label="Something went wrong.", state="error")
                    st.session_state.error = friendly_error(r)
                else:
                    ingest = r.json()
                    status.update(
                        label="Document read. Getting ready to answer questions...",
                        state="running",
                    )

                    r2 = client.post("/preprocess", progress=note)
                    if r2.status_code != 200:
                        status.update(label="Something went wrong while preparing.", state="error")
                        st.session_state.error = friendly_error(r2)
                    else:
                        prep = r2.json()
                        status.update(label="Ready!", state="complete")
                        st.session_state.doc_summary = {
                            "filename": uploaded.name,
                            "pages": ingest.get("pdf_metadata", {}).get("page_count", "?"),
                            "sections": ingest.get("total_chunks", "?"),
                            "prep_seconds": prep.get("sleep_time_sec", "?"),
                            "faq_count": prep.get("faq_count", 0),
                        }
                        st.session_state.stage = "ready"
                        remember(client)
                        time.sleep(0.4)
                        st.rerun()

            except ApiUnavailable as e:
                status.update(label="Couldn't wake the server.", state="error")
                st.session_state.error = str(e)
            except httpx.TimeoutException:
                status.update(label="Taking longer than expected.", state="error")
                st.session_state.error = (
                    "The server didn't respond in time. Free hosting can take a "
                    "minute to wake up after being idle. Please try uploading again."
                )
            except httpx.RequestError:
                status.update(label="Couldn't connect.", state="error")
                st.session_state.error = (
                    "Couldn't reach the server. Please check your connection and try again."
                )
            finally:
                remember(client)

    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Try again"):
            st.session_state.error = None
            st.session_state.api_awake = False
            st.rerun()

# ---------------------------------------------------------------------------
# Stage 2 — Ready to ask
# ---------------------------------------------------------------------------
elif st.session_state.stage == "ready":
    doc = st.session_state.doc_summary
    st.markdown('<span class="step-badge">Step 2 of 2</span>', unsafe_allow_html=True)
    st.subheader("Ask anything about your document")
    st.caption(
        f"📄 **{doc['filename']}** — {doc['pages']} pages, read in {doc['prep_seconds']}s. "
        "Ask as many questions as you like."
    )

    question = st.text_input(
        "Your question", placeholder="e.g. What is the main finding of this document?",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([1, 1])
    ask_clicked = col1.button("Ask", type="primary", use_container_width=True)
    col2.button("Upload a different document", on_click=reset_all, use_container_width=True)

    if ask_clicked and question.strip():
        client = get_client()
        with st.spinner("Thinking..."):
            try:
                r = client.post("/query", json={"question": question, "pipeline": "sleep_time"})
                if r.status_code == 200:
                    st.session_state.history.insert(0, (question, r.json(), None))
                elif r.status_code == 400:
                    # The backend restarted and lost the document it had read.
                    st.session_state.stage = "upload"
                    st.session_state.doc_summary = None
                    st.session_state.error = (
                        "The server restarted and no longer has your document. "
                        "Please upload it again."
                    )
                    remember(client)
                    st.rerun()
                else:
                    st.session_state.history.insert(0, (question, None, friendly_error(r)))
            except ApiUnavailable as e:
                st.session_state.history.insert(0, (question, None, str(e)))
            except httpx.TimeoutException:
                st.session_state.history.insert(
                    0, (question, None, "That took too long. Please try asking again.")
                )
            except httpx.RequestError:
                st.session_state.history.insert(
                    0, (question, None, "Couldn't reach the server. Please try again.")
                )
            finally:
                remember(client)

    st.write("")

    for q, result, err in st.session_state.history:
        st.markdown(f"**You asked:** {q}")
        if err:
            st.error(err)
        else:
            st.markdown(f'<div class="answer-box">{result.get("answer", "")}</div>', unsafe_allow_html=True)
            sources = (
                len(result.get("retrieved_chunks", []))
                + len(result.get("retrieved_faqs", []))
                + len(result.get("retrieved_predictions", []))
            )
            st.markdown(
                f'<p class="subtle">Answered in {result.get("latency_sec", "?")}s, '
                f"drawing on {sources} parts of the document.</p>",
                unsafe_allow_html=True,
            )
        st.divider()
