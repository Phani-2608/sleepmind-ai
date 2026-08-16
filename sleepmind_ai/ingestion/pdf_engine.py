"""PDF ingestion: extract, clean, chunk, embed, index into FAISS."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import SleepMindConfig, resolve_api_key

logger = logging.getLogger(__name__)


class PDFIngestionEngine:
    """Extracts, cleans, chunks, embeds, and indexes a research PDF."""

    def __init__(self, cfg: SleepMindConfig):
        self.cfg = cfg
        self.embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=resolve_api_key(cfg),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
        self.vectorstore: LangFAISS | None = None
        self.raw_text: str = ""
        self.chunks: list[Document] = []
        self.pdf_metadata: dict[str, Any] = {}

    def extract_text(self, pdf_path: str) -> str:
        logger.info("Extracting text from: %s", pdf_path)
        doc = fitz.open(pdf_path)
        self.pdf_metadata = {
            "title": doc.metadata.get("title", "Unknown"),
            "author": doc.metadata.get("author", "Unknown"),
            "page_count": doc.page_count,
            "file_path": pdf_path,
            "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2),
        }
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[PAGE {page_num}]\n{text}")
        doc.close()
        raw = "\n\n".join(pages)
        logger.info(
            "Extracted %s chars from %s pages.", f"{len(raw):,}", self.pdf_metadata["page_count"]
        )
        return raw

    def clean_text(self, text: str) -> str:
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\d{1,3}\n", "\n", text)
        text = re.sub(r"\n[·•–—=\-_.]{3,}\n", "\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def create_chunks(self, text: str) -> list[Document]:
        raw_chunks = self.splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(raw_chunks):
            if len(chunk.strip()) < self.cfg.min_chunk_length:
                continue
            page_match = re.search(r"\[PAGE (\d+)\]", chunk)
            page_num = int(page_match.group(1)) if page_match else -1
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": i,
                        "chunk_index": len(documents),
                        "page_number": page_num,
                        "char_count": len(chunk),
                        "word_count": len(chunk.split()),
                        **self.pdf_metadata,
                    },
                )
            )
        logger.info("Created %s document chunks.", len(documents))
        return documents

    def build_vectorstore(self, chunks: list[Document]) -> LangFAISS:
        logger.info("Building FAISS vector store (%s chunks)...", len(chunks))
        batch_size = 100
        vectorstore = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if vectorstore is None:
                vectorstore = LangFAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.add_documents(batch)
        logger.info("FAISS index built with %s vectors.", len(chunks))
        return vectorstore

    def ingest(self, pdf_path: str) -> dict[str, Any]:
        t0 = time.time()
        self.raw_text = self.extract_text(pdf_path)
        cleaned = self.clean_text(self.raw_text)
        self.chunks = self.create_chunks(cleaned)
        self.vectorstore = self.build_vectorstore(self.chunks)
        elapsed = round(time.time() - t0, 2)
        return {
            "status": "success",
            "pdf_metadata": self.pdf_metadata,
            "raw_chars": len(self.raw_text),
            "cleaned_chars": len(cleaned),
            "total_chunks": len(self.chunks),
            "avg_chunk_words": round(
                float(np.mean([c.metadata["word_count"] for c in self.chunks])), 1
            ),
            "ingestion_time_sec": elapsed,
        }

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not built. Run ingest() first.")
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def save_vectorstore(self, path: str) -> None:
        if self.vectorstore is None:
            raise RuntimeError("No vectorstore to save.")
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info("Vectorstore saved to %s", path)

    def load_vectorstore(self, path: str) -> None:
        self.vectorstore = LangFAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded from %s", path)
