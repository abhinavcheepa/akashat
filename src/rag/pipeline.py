"""RAG Pipeline Coordinator for PATHFINDER.

Coordinates document ingestion, chunking, search mode identification ('keyword search'
vs 'semantic search'), verified question generation, and QuestionManager synchronization.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.question import Question
from src.rag.document_ingestor import Chunk, DocumentIngestor
from src.rag.question_generator import QuestionGenerator
from src.rag.retriever import Retriever


class RAGPipeline:
    """Manages study material ingestion, retrieval indexing, and question generation."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_file_size_mb: float = 5.0,
        force_keyword: bool = False,
    ) -> None:
        self.ingestor = DocumentIngestor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_file_size_mb=max_file_size_mb,
        )
        self.force_keyword: bool = force_keyword

        self.chunks: List[Chunk] = []
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[QuestionGenerator] = None
        self.generated_questions: List[Question] = []

        self.loaded_file_path: str = ""
        self.loaded_file_name: str = ""
        self.active_retrieval_mode: str = "None"
        self.use_in_game: bool = True

    def process_file(
        self,
        file_path: str,
        target_question_count: int = 8,
    ) -> Tuple[bool, str, List[Question]]:
        """Ingest a study document, index chunks, and generate questions.

        Returns:
            Tuple of (success: bool, status_message: str, questions: List[Question]).
        """
        # 1. Ingestion & Chunking
        success, msg, chunks = self.ingestor.ingest_file(file_path)
        if not success or not chunks:
            return False, msg, []

        self.chunks = chunks
        self.loaded_file_path = file_path
        self.loaded_file_name = chunks[0].source_file

        # 2. Retrieval Engine Initialization
        self.retriever = Retriever(self.chunks, force_keyword=self.force_keyword)
        self.active_retrieval_mode = self.retriever.active_mode

        # 3. Question Generation
        self.generator = QuestionGenerator(self.chunks)
        self.generated_questions = self.generator.generate_all(target_count=target_question_count)

        if not self.generated_questions:
            return False, "Ingested document successfully, but could not extract enough grounded questions.", []

        return (
            True,
            f"Loaded '{self.loaded_file_name}' ({len(self.chunks)} chunks). "
            f"Active engine: {self.active_retrieval_mode}. "
            f"Generated {len(self.generated_questions)} verified questions.",
            self.generated_questions,
        )

    def get_status_summary(self) -> Dict[str, Any]:
        """Return human-readable metadata for console UI."""
        return {
            "loaded_file": self.loaded_file_name or "No material uploaded",
            "chunk_count": len(self.chunks),
            "retrieval_mode": self.active_retrieval_mode,
            "questions_count": len(self.generated_questions),
            "use_in_game": self.use_in_game and len(self.generated_questions) > 0,
        }
