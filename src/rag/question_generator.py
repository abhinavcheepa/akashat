"""Question Generation and Verification for PATHFINDER RAG pipeline.

Generates verified 4-option Question entities from document chunks:
1. Offline Cloze Engine: Extracts salient concept sentences, creates cloze blanks,
   and selects plausible distractors from the ingested document terms.
2. Optional LLM Engine: Connects to OpenAI or Anthropic API only when an API key
   is detected in environment variables (never files, never logged).
3. Rigorous Validator: Enforces 4 distinct options, in-range answer index, non-empty stems,
   and verifies that the answer text is grounded in the source chunk.
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.question import Question
from src.rag.document_ingestor import Chunk


class QuestionValidator:
    """Strictly validates generated questions against grounding and formatting rules."""

    @staticmethod
    def validate(question: Question, source_chunk: Chunk) -> bool:
        """Validate question integrity. Drops any invalid questions.

        Rules:
        - Stem text must not be blank and must contain meaningful content
        - Exactly 4 options
        - All 4 options must be distinct (no duplicates)
        - Answer index must be integer 0..3
        - The answer text must actually be grounded/present in the source chunk
        """
        # 1. Stem validation
        stem = question.question.strip()
        if not stem or len(stem) < 10:
            return False

        # 2. Options count and length
        if len(question.options) != 4:
            return False

        cleaned_options = [opt.strip() for opt in question.options]
        for opt in cleaned_options:
            if not opt:
                return False

        # 3. No duplicate options (case-insensitive)
        lower_options = [opt.lower() for opt in cleaned_options]
        if len(set(lower_options)) != 4:
            return False

        # 4. Answer index range
        if not isinstance(question.answer, int) or not (0 <= question.answer < 4):
            return False

        # 5. Grounding check: answer text must be supported by the source chunk
        ans_text = cleaned_options[question.answer].lower()
        chunk_text_lower = source_chunk.text.lower()

        # Check if full answer phrase or individual salient tokens appear in source chunk
        if ans_text not in chunk_text_lower:
            tokens = [t for t in re.findall(r"\b\w{3,}\b", ans_text)]
            if not tokens or not any(t in chunk_text_lower for t in tokens):
                return False

        return True


class OfflineClozeGenerator:
    """Generates cloze concept questions from text chunks with local document distractors."""

    STOPWORDS = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his",
        "from", "they", "say", "her", "she", "will", "one", "all", "would", "there",
        "their", "what", "out", "about", "who", "get", "which", "when", "make", "can",
        "like", "time", "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
        "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "been",
    }

    def __init__(self, all_chunks: List[Chunk]) -> None:
        self.all_chunks: List[Chunk] = all_chunks
        self.term_bank: List[str] = self._extract_global_terms()

    def _extract_global_terms(self) -> List[str]:
        """Harvest capitalized or technical candidate terms across all chunks for distractors."""
        terms: Set[str] = set()
        for chunk in self.all_chunks:
            # Match capitalized phrases or alphanumeric acronyms/words >= 4 chars
            candidates = re.findall(r"\b[A-Za-z0-9_]{4,}\b", chunk.text)
            for cand in candidates:
                if cand.lower() not in self.STOPWORDS and not cand.isdigit():
                    terms.add(cand)
        return list(terms)

    def generate_questions(self, target_count: int = 5) -> List[Question]:
        """Generate verified cloze questions from chunks."""
        questions: List[Question] = []
        if not self.all_chunks or len(self.term_bank) < 4:
            return []

        # Iterate over chunks and sentences
        random.seed(42)
        shuffled_chunks = list(self.all_chunks)
        random.shuffle(shuffled_chunks)

        q_counter = 1
        for chunk in shuffled_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 35 or len(sent) > 220:
                    continue

                # Find candidate key terms in this sentence
                words = re.findall(r"\b[A-Za-z0-9_]{4,}\b", sent)
                valid_keywords = [
                    w for w in words
                    if w.lower() not in self.STOPWORDS and not w.isdigit()
                ]
                if not valid_keywords:
                    continue

                # Prefer capitalized or distinct terms
                target_word = max(valid_keywords, key=lambda w: (w[0].isupper(), len(w)))

                # Mask term with a blank
                cloze_pattern = re.compile(re.escape(target_word), re.IGNORECASE)
                masked_stem = cloze_pattern.sub("________", sent, count=1)
                question_text = f"Fill in the blank from your study material:\n\"{masked_stem}\""

                # Select 3 distractors from term bank
                candidate_distractors = [
                    t for t in self.term_bank
                    if t.lower() != target_word.lower() and len(t) >= 3
                ]
                if len(candidate_distractors) < 3:
                    continue

                distractors = random.sample(candidate_distractors, 3)
                options = [target_word] + distractors
                random.shuffle(options)
                ans_idx = options.index(target_word)

                source_ref = f"{chunk.source_file} [{chunk.section}] (Chunk {chunk.chunk_id})"
                explanation = f"Auto-generated from your material: '{sent}'. Source: {source_ref}."

                q = Question(
                    id=f"rag_auto_{q_counter}",
                    topic="Custom Materials",
                    difficulty="Medium",
                    question=question_text,
                    options=options,
                    answer=ans_idx,
                    explanation=explanation,
                )

                if QuestionValidator.validate(q, chunk):
                    questions.append(q)
                    q_counter += 1
                    if len(questions) >= target_count:
                        return questions

        return questions


class LLMQuestionGenerator:
    """Generates grounded questions using an external LLM when an API key is in env."""

    def __init__(self, api_key: str, provider: str = "openai") -> None:
        self.api_key: str = api_key
        self.provider: str = provider

    def generate_from_chunk(self, chunk: Chunk) -> Optional[Question]:
        """Call external LLM via standard urllib.request with strict chunk grounding.

        Mockable in unit tests without making real network requests.
        """
        prompt = (
            f"You are creating a multiple-choice question for an educational game.\n"
            f"Use ONLY the following text from {chunk.source_file}:\n"
            f"\"\"\"\n{chunk.text}\n\"\"\"\n"
            f"Generate 1 question with exactly 4 options where 1 is correct and 3 are incorrect distractors.\n"
            f"Return strictly valid JSON with keys: 'question', 'options' (array of 4 strings), 'answer' (integer 0-3), 'explanation'."
        )

        try:
            # We use urllib to avoid external dependencies like openai or anthropic
            import urllib.request
            import urllib.error

            if self.provider == "openai":
                req_url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                }
            else:
                req_url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                }

            req = urllib.request.Request(
                req_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            # Timeout after 8 seconds
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            content = (
                data["choices"][0]["message"]["content"]
                if self.provider == "openai"
                else data["content"][0]["text"]
            )

            # Extract JSON block
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return None

            parsed = json.loads(json_match.group(0))
            source_ref = f"{chunk.source_file} [{chunk.section}] (Chunk {chunk.chunk_id})"
            q = Question(
                id=f"rag_llm_{chunk.chunk_id}",
                topic="Custom Materials",
                difficulty="Medium",
                question=parsed.get("question", ""),
                options=parsed.get("options", []),
                answer=parsed.get("answer", 0),
                explanation=f"{parsed.get('explanation', '')} [Source: {source_ref}]",
            )

            if QuestionValidator.validate(q, chunk):
                return q
            return None

        except Exception:
            # Fall back cleanly without crash
            return None


class QuestionGenerator:
    """Coordinates offline cloze generation and optional environment-keyed LLM generation."""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks: List[Chunk] = chunks
        self.cloze_gen = OfflineClozeGenerator(chunks)
        self.llm_gen: Optional[LLMQuestionGenerator] = self._init_llm_if_available()

    def _init_llm_if_available(self) -> Optional[LLMQuestionGenerator]:
        """Detect API keys strictly in environment variables (never files)."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            return LLMQuestionGenerator(api_key=openai_key.strip(), provider="openai")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.strip():
            return LLMQuestionGenerator(api_key=anthropic_key.strip(), provider="anthropic")

        return None

    def generate_all(self, target_count: int = 8) -> List[Question]:
        """Generate verified questions using available engines."""
        questions: List[Question] = []

        # 1. If LLM generator is active, attempt to generate questions
        if self.llm_gen and self.chunks:
            for chunk in self.chunks[:target_count]:
                q = self.llm_gen.generate_from_chunk(chunk)
                if q:
                    questions.append(q)
                if len(questions) >= target_count:
                    return questions

        # 2. Fill remaining or all quota using the offline cloze generator
        needed = target_count - len(questions)
        if needed > 0:
            offline_qs = self.cloze_gen.generate_questions(target_count=needed)
            questions.extend(offline_qs)

        return questions
