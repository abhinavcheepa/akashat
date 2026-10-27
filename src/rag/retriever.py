"""Retrieval engine for PATHFINDER.

Default: pure-Python BM25 keyword search with zero external dependencies.
Labeled strictly as 'keyword search' across the codebase and user interface.
Optional upgrade: activates 'semantic search' using sentence-transformers and FAISS
only if those optional packages are installed and initialized.
"""

from collections import Counter
import math
import re
from typing import Any, List, Optional, Tuple

from src.rag.document_ingestor import Chunk


class BM25KeywordRetriever:
    """Pure-Python BM25 keyword search engine with no external libraries."""

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: List[Chunk] = chunks
        self.k1: float = k1
        self.b: float = b

        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Counter[str] = Counter()
        self.doc_tokens: List[List[str]] = []
        self.num_docs: int = len(chunks)

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric keywords."""
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]

    def _build_index(self) -> None:
        """Construct inverted index and document length stats."""
        total_len = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            l = len(tokens)
            self.doc_len.append(l)
            total_len += l

            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.avg_doc_len = (total_len / float(self.num_docs)) if self.num_docs > 0 else 0.0

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Rank chunks by BM25 relevance score."""
        if self.num_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(c, 0.0) for c in self.chunks[:top_k]]

        scores: List[float] = [0.0] * self.num_docs

        for term in query_tokens:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue

            # Standard BM25 IDF formulation
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue

                d_len = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_idx] += score

        ranked_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], scores[i]) for i in ranked_indices[:top_k]]


class SemanticRetriever:
    """Optional embedding-based search using sentence-transformers and FAISS."""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks: List[Chunk] = chunks
        self.model = None
        self.index = None
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        import faiss
        import numpy as np

        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results


class Retriever:
    """Unified retrieval interface exposing the active search mode honestly."""

    def __init__(self, chunks: List[Chunk], force_keyword: bool = False) -> None:
        self.chunks: List[Chunk] = chunks
        self.active_mode: str = "keyword search"
        self._retriever_impl = None

        if not force_keyword and self._can_use_semantic():
            try:
                self._retriever_impl = SemanticRetriever(chunks)
                self.active_mode = "semantic search"
            except Exception:
                # Fall back to keyword search
                self._retriever_impl = BM25KeywordRetriever(chunks)
                self.active_mode = "keyword search"
        else:
            self._retriever_impl = BM25KeywordRetriever(chunks)
            self.active_mode = "keyword search"

    def _can_use_semantic(self) -> bool:
        """Check if sentence-transformers and faiss are available."""
        try:
            import sentence_transformers
            import faiss
            return True
        except ImportError:
            return False

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Retrieve most relevant chunks for query using active search engine."""
        if not self._retriever_impl or not self.chunks:
            return []
        return self._retriever_impl.search(query, top_k=top_k)
