"""Optional RAG package for PATHFINDER.

Provides document ingestion, chunking, BM25 keyword search (or optional semantic embeddings),
and grounded question generation.
"""

from src.rag.document_ingestor import Chunk, DocumentIngestor
from src.rag.pipeline import RAGPipeline
from src.rag.question_generator import QuestionGenerator, QuestionValidator
from src.rag.retriever import BM25KeywordRetriever, Retriever

__all__ = [
    "Chunk",
    "DocumentIngestor",
    "BM25KeywordRetriever",
    "Retriever",
    "QuestionValidator",
    "QuestionGenerator",
    "RAGPipeline",
]
