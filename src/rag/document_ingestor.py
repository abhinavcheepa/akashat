"""Document Ingestion and Text Chunking for PATHFINDER.

Accepts TXT and Markdown files natively with zero external dependencies.
Gracefully detects PDF support via pypdf with fallback warning and no crashes.
Handles oversized files, empty documents, unsupported types, and multiple encodings.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple


class Chunk:
    """Represents a discrete segment of ingested study material.

    Attributes:
        chunk_id: Sequential index identifier (e.g. 'c1', 'c2').
        source_file: Original filename.
        section: Section header or page number reference.
        text: Normalized text content of the chunk.
        token_count: Approximate word count.
    """

    def __init__(
        self,
        chunk_id: str,
        source_file: str,
        text: str,
        section: str = "General",
    ) -> None:
        self.chunk_id: str = chunk_id
        self.source_file: str = source_file
        self.text: str = text.strip()
        self.section: str = section
        self.token_count: int = len(self.text.split())

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, file={self.source_file}, words={self.token_count})"


class DocumentIngestor:
    """Ingests and splits study documents into structured chunks with guaranteed overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_file_size_mb: float = 5.0,
    ) -> None:
        self.chunk_size: int = max(60, chunk_size)
        self.chunk_overlap: int = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.max_file_bytes: int = int(max_file_size_mb * 1024 * 1024)

    def ingest_file(self, file_path: str) -> Tuple[bool, str, List[Chunk]]:
        """Ingest document file and generate chunks.

        Returns:
            Tuple of (success: bool, status_message: str, chunks: List[Chunk]).
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}", []

        # 1. File size check
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "File is empty (0 bytes).", []
        if size > self.max_file_bytes:
            mb = size / (1024 * 1024)
            return False, f"File exceeds maximum size limit ({mb:.2f} MB > {self.max_file_bytes / (1024*1024):.1f} MB).", []

        # 2. File extension handling
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        filename = os.path.basename(file_path)

        if ext in (".txt", ".md", ".markdown"):
            text, err = self._read_text_file(file_path)
            if err:
                return False, err, []
            clean_text = self._clean_text(text)
            if not clean_text:
                return False, "File contains no readable text content.", []
            chunks = self._chunk_continuous_text(clean_text, filename, default_section="Document")

        elif ext == ".pdf":
            success, msg, pdf_sections = self._read_pdf_file(file_path)
            if not success:
                return False, msg, []
            chunks = []
            chunk_counter = 1
            for page_label, page_text in pdf_sections:
                clean_page = self._clean_text(page_text)
                if clean_page:
                    page_chunks = self._chunk_continuous_text(
                        clean_page, filename, default_section=page_label, start_idx=chunk_counter
                    )
                    chunks.extend(page_chunks)
                    chunk_counter += len(page_chunks)
        else:
            return False, f"Unsupported file format '{ext}'. Supported types: .txt, .md, .pdf", []

        if not chunks:
            return False, "Document contains no readable textual content after normalization.", []

        return True, f"Successfully ingested {len(chunks)} chunks from {filename}.", chunks

    def _read_text_file(self, file_path: str) -> Tuple[str, Optional[str]]:
        """Read text with fallback across encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read(), None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return "", f"Error reading file: {e}"
        return "", "Unable to decode file with standard text encodings."

    def _read_pdf_file(self, file_path: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
        """Extract text from PDF using pypdf if installed."""
        try:
            import pypdf
        except ImportError:
            return False, "PDF support needs: pip install pypdf", []

        try:
            reader = pypdf.PdfReader(file_path)
            sections: List[Tuple[str, str]] = []
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append((f"Page {page_idx}", page_text))

            if not sections:
                return False, "PDF contains no extractable text (may be image scans).", []

            return True, "", sections
        except Exception as e:
            return False, f"Failed parsing PDF: {e}", []

    def _clean_text(self, text: str) -> str:
        """Normalize whitespaces and control characters."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_continuous_text(
        self,
        text: str,
        source_file: str,
        default_section: str = "General",
        start_idx: int = 1,
    ) -> List[Chunk]:
        """Slice text into overlapping chunks using sliding word windows.

        Guarantees consecutive chunks share overlapping context words.
        """
        words = text.split()
        if not words:
            return []

        # Average word length is ~5.5 chars; calculate word count per chunk
        target_words = max(8, self.chunk_size // 6)
        overlap_words = max(2, min(self.chunk_overlap // 6, target_words // 2))

        chunks: List[Chunk] = []
        start = 0
        idx = start_idx

        while start < len(words):
            end = min(start + target_words, len(words))
            chunk_slice = words[start:end]
            chunk_str = " ".join(chunk_slice)

            # Try to identify header if present in slice
            section = default_section
            for w in chunk_slice:
                if w.startswith("#"):
                    section = w.lstrip("#").strip() or default_section
                    break

            chunks.append(Chunk(f"c{idx}", source_file, chunk_str, section))
            idx += 1

            if end >= len(words):
                break

            # Step forward by (target_words - overlap_words) to guarantee overlap
            step = max(1, target_words - overlap_words)
            start += step

        return chunks
