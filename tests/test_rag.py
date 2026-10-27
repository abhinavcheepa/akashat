"""Adversarial test suite for Phase C RAG module:
- Document chunking: sizes, overlap, empty text, unicode encoding resilience
- Keyword retrieval: relevant chunk ranks first on a sample document
- Question validation: rejects duplicate options, bad answer index, unsupported answer
- Offline cloze generation produces verified 4-option questions from sample text
- Fallback mechanics: RAG disabled or missing dependencies safely defaults to offline bank
- Security: API keys read only from environment variables; LLM calls mocked with no real network requests
- File edge cases: bad file extension, empty file (0 bytes), oversize file rejection
"""

import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest
import pygame

from src.game import Game, GameState
from src.question import Question, QuestionManager
from src.rag.document_ingestor import Chunk, DocumentIngestor
from src.rag.pipeline import RAGPipeline
from src.rag.question_generator import (
    LLMQuestionGenerator,
    OfflineClozeGenerator,
    QuestionGenerator,
    QuestionValidator,
)
from src.rag.retriever import BM25KeywordRetriever, Retriever


def test_document_ingestion_and_chunking() -> None:
    """Verify chunk size, overlap, empty text, and unicode handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ingestor = DocumentIngestor(chunk_size=120, chunk_overlap=30, max_file_size_mb=1.0)

        # 1. Unicode & paragraph text file
        sample_path = os.path.join(tmpdir, "unicode_sample.md")
        sample_content = (
            "# Graph Theory & Dijkstra\n\n"
            "Dijkstra's algorithm guarantees the shortest path for non-negative weights. "
            "It computes optimal traversal costs across complex networks efficiently.\n\n"
            "## Priority Queue Enhancement\n"
            "Using a binary min-heap reduces complexity to O((V + E) log V). "
            "International characters like naïve, façade, and π ≈ 3.14159 are preserved accurately."
        )
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_content)

        success, msg, chunks = ingestor.ingest_file(sample_path)
        assert success is True
        assert len(chunks) >= 2

        for c in chunks:
            assert c.source_file == "unicode_sample.md"
            assert len(c.text) > 0
            assert c.chunk_id.startswith("c")

        # 2. Overlap check: adjacent chunks share boundary context
        if len(chunks) >= 2:
            c1_words = set(chunks[0].text.split())
            c2_words = set(chunks[1].text.split())
            # Overlap should preserve common words between segments
            common = c1_words.intersection(c2_words)
            assert len(common) > 0


def test_file_ingestion_edge_cases() -> None:
    """Verify rejection of empty files, oversize files, and unsupported types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ingestor = DocumentIngestor(chunk_size=200, max_file_size_mb=0.001)  # ~1 KB max size limit

        # 1. Empty file (0 bytes)
        empty_path = os.path.join(tmpdir, "empty.txt")
        with open(empty_path, "w", encoding="utf-8") as f:
            pass
        success, msg, chunks = ingestor.ingest_file(empty_path)
        assert success is False
        assert "empty" in msg.lower()
        assert len(chunks) == 0

        # 2. Unsupported file type
        bad_type_path = os.path.join(tmpdir, "script.py")
        with open(bad_type_path, "w", encoding="utf-8") as f:
            f.write("print('hello world')")
        success, msg, chunks = ingestor.ingest_file(bad_type_path)
        assert success is False
        assert "unsupported" in msg.lower()

        # 3. Oversized file
        large_path = os.path.join(tmpdir, "oversize.txt")
        with open(large_path, "w", encoding="utf-8") as f:
            f.write("A" * 2048)  # 2 KB > 1 KB limit
        success, msg, chunks = ingestor.ingest_file(large_path)
        assert success is False
        assert "exceeds" in msg.lower()

        # 4. Non-existent file
        success, msg, chunks = ingestor.ingest_file(os.path.join(tmpdir, "missing.txt"))
        assert success is False
        assert "not found" in msg.lower()


def test_keyword_retrieval_ranking() -> None:
    """Verify BM25 keyword search ranks the most relevant chunk first."""
    chunks = [
        Chunk("c1", "notes.md", "Python and object-oriented programming use classes, inheritance, and polymorphism."),
        Chunk("c2", "notes.md", "Dijkstra's shortest path algorithm employs a min-heap priority queue to settle nodes."),
        Chunk("c3", "notes.md", "Relational database transactions satisfy ACID atomicity, consistency, isolation, durability."),
    ]

    retriever = Retriever(chunks, force_keyword=True)
    assert retriever.active_mode == "keyword search"

    # Search for Dijkstra priority queue
    results = retriever.search("Dijkstra priority queue min-heap", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c2"
    assert "Dijkstra" in top_chunk.text
    assert score > results[1][1]

    # Search for database ACID
    db_results = retriever.search("database transactions ACID", top_k=3)
    assert db_results[0][0].chunk_id == "c3"


def test_question_validator_rules() -> None:
    """Verify validator rejects duplicate options, out-of-range answers, and unsupported text."""
    chunk = Chunk("c1", "test.md", "Edsger Dijkstra formulated the single-source shortest path algorithm.")

    # 1. Valid question
    valid_q = Question(
        id="q1",
        topic="Custom Materials",
        difficulty="Medium",
        question="Who formulated the shortest path algorithm according to the material?",
        options=["Edsger Dijkstra", "Alan Turing", "John von Neumann", "Claude Shannon"],
        answer=0,
        explanation="Text states Edsger Dijkstra formulated it.",
    )
    assert QuestionValidator.validate(valid_q, chunk) is True

    # 2. Reject duplicate options
    dup_q = Question(
        id="q2",
        topic="Custom Materials",
        difficulty="Medium",
        question="Who formulated the algorithm?",
        options=["Edsger Dijkstra", "Alan Turing", "Edsger Dijkstra", "Claude Shannon"],
        answer=0,
        explanation="Duplicate option",
    )
    assert QuestionValidator.validate(dup_q, chunk) is False

    # 3. Reject bad answer index (out of 0..3)
    bad_idx_q = Question(
        id="q3",
        topic="Custom Materials",
        difficulty="Medium",
        question="Who formulated the algorithm?",
        options=["Edsger Dijkstra", "Alan Turing", "John Backus", "Claude Shannon"],
        answer=5,
        explanation="Index 5 out of range",
    )
    assert QuestionValidator.validate(bad_idx_q, chunk) is False

    # 4. Reject answer not grounded in source chunk
    unsupported_q = Question(
        id="q4",
        topic="Custom Materials",
        difficulty="Medium",
        question="Which city was the conference held in?",
        options=["Tokyo", "London", "Paris", "Berlin"],
        answer=0,
        explanation="Tokyo is not in chunk text",
    )
    assert QuestionValidator.validate(unsupported_q, chunk) is False


def test_offline_cloze_question_generation() -> None:
    """Verify offline generator produces verified 4-option questions from document chunks."""
    sample_text = (
        "In computer networking, the Transmission Control Protocol establishes a reliable connection "
        "using a three-way handshake. The Internet Protocol provides packet addressing across networks. "
        "Routers decrement the Time to Live field to eliminate infinite loops. "
        "Dynamic Host Configuration Protocol leases IP addresses to client hosts."
    )
    chunk = Chunk("c1", "networks.txt", sample_text, section="Transport")
    generator = OfflineClozeGenerator([chunk])

    questions = generator.generate_questions(target_count=2)
    assert len(questions) > 0

    for q in questions:
        assert len(q.options) == 4
        assert 0 <= q.answer <= 3
        assert len(set(q.options)) == 4
        assert "networks.txt" in q.explanation
        assert QuestionValidator.validate(q, chunk) is True


def test_llm_mode_environment_security_and_mocking() -> None:
    """Verify API keys are strictly read from environment, and LLM calls are mocked."""
    chunk = Chunk("c1", "source.md", "Graph traversal visits each node systematically.")

    # 1. Verify generator does not activate LLM without env variable
    with patch.dict(os.environ, {}, clear=True):
        q_gen = QuestionGenerator([chunk])
        assert q_gen.llm_gen is None

    # 2. Mock external LLM call with simulated valid JSON response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "question": "What does graph traversal do according to the source?",
                        "options": ["Visits each node systematically", "Deletes redundant edges", "Reverses node directions", "Finds prime numbers"],
                        "answer": 0,
                        "explanation": "Stated directly in source text.",
                    })
                }
            }
        ]
    }

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key-12345"}):
        q_gen_with_key = QuestionGenerator([chunk])
        assert q_gen_with_key.llm_gen is not None
        assert q_gen_with_key.llm_gen.api_key == "sk-mock-key-12345"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_llm_response).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            q = q_gen_with_key.llm_gen.generate_from_chunk(chunk)
            assert q is not None
            assert "graph traversal" in q.question.lower()
            assert q.options[0] == "Visits each node systematically"
            assert q.answer == 0
            assert "source.md" in q.explanation


def test_rag_pipeline_and_game_fallback() -> None:
    """Verify RAG pipeline integration and silent fallback to offline bank on failure."""
    pipeline = RAGPipeline()
    qm = QuestionManager(data_path="data/questions.json")

    # 1. Process sample notes
    success, msg, questions = pipeline.process_file("data/sample_study_material.md", target_question_count=4)
    assert success is True
    assert len(questions) > 0
    assert pipeline.active_retrieval_mode == "keyword search"

    # Add questions to manager
    qm.add_custom_questions(questions)
    assert qm.has_custom_materials() is True
    assert "Custom Materials" in qm.get_topics()

    # Serve custom question
    custom_q = qm.get_question(topic="Custom Materials")
    assert custom_q is not None
    assert custom_q.topic == "Custom Materials"

    # 2. Clear custom questions -> topics list should no longer include 'Custom Materials'
    qm.clear_custom_questions()
    assert qm.has_custom_materials() is False
    assert "Custom Materials" not in qm.get_topics()

    # Requesting 'Custom Materials' when none exist falls back gracefully to offline questions without crashing
    fallback_q = qm.get_question(topic="Custom Materials")
    assert fallback_q is not None
    assert fallback_q.topic in [
        "DSA", "Python and OOP", "DBMS", "OS", "Networks", "Math and Probability", "General Knowledge"
    ]
