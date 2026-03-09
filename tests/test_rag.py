"""
Tests for the RAG pipeline — chunker, ingestor, retrieval tool.

Strategy: test each layer in isolation with mocked or in-memory dependencies.
No real embeddings or ChromaDB required — Embedder and VectorStore are mocked
or replaced with thin in-memory stubs.

Covers:
  Chunker:
    - Short text → single chunk
    - Multi-paragraph text → one chunk per paragraph
    - Long paragraph → sub-chunked at sentence boundaries
    - Empty text → empty list
    - source and chunk_index preserved

  Ingestor:
    - ingest_text returns chunk count
    - Unsupported suffix raises ValueError
    - ingest_file reads and ingests correctly
    - ingest_upload uses original_name for type detection and source label
    - Re-ingesting same source is safe (upsert)

  RetrievalTool:
    - Empty store → success with empty results and empty message
    - Results above threshold → returned with citations
    - Results below threshold → empty results, below_threshold flag set
    - Mixed hits → only above-threshold returned
    - metadata retrieval_event always present
    - source deduplication
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auri.rag.chunker import chunk_text
from auri.rag.ingest import Ingestor, classify_file_type
from auri.rag.retriever import Retriever
from auri.tools.retrieval import RetrievalTool


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_single_short_paragraph():
    chunks = chunk_text("Hello world.", source="test.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world."
    assert chunks[0]["source"] == "test.txt"
    assert chunks[0]["chunk_index"] == "0"


def test_two_paragraphs_produce_two_chunks():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_text(text, source="doc.md")
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == "0"
    assert chunks[1]["chunk_index"] == "1"


def test_empty_text_returns_empty_list():
    assert chunk_text("", source="empty.txt") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n   ", source="blank.txt") == []


def test_source_preserved_in_all_chunks():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, source="my_source.py")
    assert all(c["source"] == "my_source.py" for c in chunks)


def test_long_paragraph_sub_chunked():
    # Build a paragraph of 5 sentences, each ~250 chars, total > 800
    sentence = "A" * 249 + "."
    text = " ".join([sentence] * 5)
    chunks = chunk_text(text, source="long.txt", max_chars=800)
    assert len(chunks) > 1
    # Sub-chunks have dotted index
    assert "." in chunks[0]["chunk_index"]


def test_max_chars_respected():
    sentence = "Word " * 40 + "."  # ~200 chars
    # 6 sentences → ~1200 chars in one paragraph; max_chars=300
    text = " ".join([sentence] * 6)
    chunks = chunk_text(text, source="x.txt", max_chars=300)
    for c in chunks:
        # Each chunk should be at most a bit over max_chars (sentence boundary)
        assert len(c["text"]) < 300 * 3  # generous upper bound


def test_chunk_index_string():
    chunks = chunk_text("Para.\n\nPara.", source="f.txt")
    for c in chunks:
        assert isinstance(c["chunk_index"], str)


# ── classify_file_type ────────────────────────────────────────────────────────

def test_classify_python_is_code():
    assert classify_file_type(Path("foo.py")) == "code"


def test_classify_markdown_is_docs():
    assert classify_file_type(Path("README.md")) == "docs"


def test_classify_yaml_is_config():
    assert classify_file_type(Path("config.yaml")) == "config"


def test_classify_unknown_is_other():
    assert classify_file_type(Path("foo.xyz")) == "other"


# ── Ingestor ──────────────────────────────────────────────────────────────────

def make_ingestor():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]  # one fake embedding per call
    def embed_side_effect(texts):
        return [[0.1] * 10 for _ in texts]
    embedder.embed.side_effect = embed_side_effect

    store = MagicMock()
    return Ingestor(embedder=embedder, store=store), embedder, store


def test_ingest_text_returns_chunk_count():
    ingestor, embedder, store = make_ingestor()
    count = ingestor.ingest_text("Para one.\n\nPara two.", source="doc.md")
    assert count == 2
    assert store.add.called


def test_ingest_text_empty_returns_zero():
    ingestor, _, store = make_ingestor()
    count = ingestor.ingest_text("", source="empty.txt")
    assert count == 0
    store.add.assert_not_called()


def test_ingest_file_reads_content(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Hello.\n\nWorld.")
    ingestor, _, store = make_ingestor()
    count = ingestor.ingest_file(f)
    assert count == 2
    assert store.add.called


def test_ingest_file_unsupported_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    ingestor, _, _ = make_ingestor()
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestor.ingest_file(f)


def test_ingest_upload_uses_original_name(tmp_path):
    # File stored with UUID name but we pass the original name
    tmp_file = tmp_path / "abc123-uuid"
    tmp_file.write_text("Content here.")
    ingestor, _, store = make_ingestor()
    count = ingestor.ingest_upload(tmp_file, original_name="report.txt")
    assert count >= 1
    # source label should be the original name, not the tmp path
    call_args = store.add.call_args[0]
    chunks = call_args[0]
    assert all(c["source"] == "report.txt" for c in chunks)


def test_ingest_upload_unsupported_raises(tmp_path):
    f = tmp_path / "data"
    f.write_bytes(b"binary")
    ingestor, _, _ = make_ingestor()
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestor.ingest_upload(f, original_name="data.png")


def test_ingest_upload_no_suffix_raises(tmp_path):
    f = tmp_path / "nosuffix"
    f.write_text("data")
    ingestor, _, _ = make_ingestor()
    with pytest.raises(ValueError, match="Cannot determine file type"):
        ingestor.ingest_upload(f, original_name="nosuffix")


def test_reingest_same_source_calls_upsert(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Same content.")
    ingestor, _, store = make_ingestor()
    ingestor.ingest_file(f)
    ingestor.ingest_file(f)
    # store.add should be called twice (upsert handles dedup inside store)
    assert store.add.call_count == 2


# ── RetrievalTool ─────────────────────────────────────────────────────────────

def make_retriever(hits: list[dict] | None = None, empty: bool = False) -> Retriever:
    retriever = MagicMock(spec=Retriever)
    retriever.is_empty.return_value = empty
    retriever.retrieve.return_value = hits or []
    return retriever


def run(coro):
    return asyncio.run(coro)


def test_empty_store_returns_empty_results():
    tool = RetrievalTool(make_retriever(empty=True))
    result = run(tool.run(query="anything"))
    assert result.success is True
    data = json.loads(result.to_json())
    assert data["output"]["results"] == []
    ev = result.metadata["retrieval_event"]
    assert ev["chunks_returned"] == 0
    assert ev["below_threshold"] is False  # not below threshold — just empty


def test_hits_above_threshold_returned():
    hits = [
        {"text": "relevant chunk", "source": "doc.md", "score": 0.80},
        {"text": "another chunk", "source": "doc.md", "score": 0.60},
    ]
    tool = RetrievalTool(make_retriever(hits=hits))
    result = run(tool.run(query="test query"))
    data = json.loads(result.to_json())
    assert len(data["output"]["results"]) == 2
    ev = result.metadata["retrieval_event"]
    assert ev["chunks_returned"] == 2
    assert ev["below_threshold"] is False
    assert ev["top_score"] == 0.80


def test_hits_below_threshold_returns_empty():
    hits = [
        {"text": "weak chunk", "source": "doc.md", "score": 0.20},
    ]
    tool = RetrievalTool(make_retriever(hits=hits))
    result = run(tool.run(query="obscure query"))
    data = json.loads(result.to_json())
    assert data["output"]["results"] == []
    ev = result.metadata["retrieval_event"]
    assert ev["below_threshold"] is True
    assert ev["top_score"] == 0.20
    assert ev["chunks_returned"] == 0


def test_mixed_hits_only_strong_returned():
    hits = [
        {"text": "strong", "source": "a.md", "score": 0.90},
        {"text": "weak",   "source": "b.md", "score": 0.10},
    ]
    tool = RetrievalTool(make_retriever(hits=hits))
    result = run(tool.run(query="query"))
    data = json.loads(result.to_json())
    assert len(data["output"]["results"]) == 1
    assert data["output"]["results"][0]["snippet"] == "strong"


def test_source_deduplication():
    hits = [
        {"text": "chunk 1", "source": "doc.md", "score": 0.90},
        {"text": "chunk 2", "source": "doc.md", "score": 0.80},
        {"text": "chunk 3", "source": "other.md", "score": 0.70},
    ]
    tool = RetrievalTool(make_retriever(hits=hits))
    result = run(tool.run(query="q"))
    ev = result.metadata["retrieval_event"]
    # doc.md appears twice but should only be listed once
    assert ev["sources"].count("doc.md") == 1
    assert "other.md" in ev["sources"]


def test_retrieval_event_always_in_metadata():
    # Empty store path
    tool = RetrievalTool(make_retriever(empty=True))
    result = run(tool.run(query="q"))
    assert "retrieval_event" in result.metadata

    # Below-threshold path
    tool2 = RetrievalTool(make_retriever(hits=[{"text": "x", "source": "f", "score": 0.1}]))
    result2 = run(tool2.run(query="q"))
    assert "retrieval_event" in result2.metadata

    # Hit path
    tool3 = RetrievalTool(make_retriever(hits=[{"text": "x", "source": "f", "score": 0.9}]))
    result3 = run(tool3.run(query="q"))
    assert "retrieval_event" in result3.metadata


def test_result_contains_snippet_source_relevance():
    hits = [{"text": "content", "source": "notes.md", "score": 0.75}]
    tool = RetrievalTool(make_retriever(hits=hits))
    result = run(tool.run(query="q"))
    data = json.loads(result.to_json())
    item = data["output"]["results"][0]
    assert item["snippet"] == "content"
    assert item["source"] == "notes.md"
    assert item["relevance"] == 0.75
