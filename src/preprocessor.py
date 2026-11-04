"""Text preprocessing, token budgeting, and semantic pre-matching utilities.

Cleans and normalizes unstructured resume and job description text,
estimates token consumption to optimize prompt payload, and computes
lexical/semantic similarity using scikit-learn TF-IDF and cosine similarity.
"""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# pylint: disable=too-few-public-methods


@dataclass
class PreMatchResult:
    """Container for preliminary semantic analysis between JD and Resume."""

    similarity_score: float
    shared_keywords: List[str] = field(default_factory=list)
    resume_token_estimate: int = 0
    jd_token_estimate: int = 0


class TextPreprocessor:
    """Text normalization and optimization engine."""

    # Regex patterns for cleaning
    _PAGE_NUM_PATTERN = re.compile(
        r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+)$", re.IGNORECASE | re.MULTILINE
    )
    _REPEATED_SYMBOLS_PATTERN = re.compile(r"[-=_~*]{3,}")
    _BULLET_PATTERN = re.compile(
        r"^[\s\t]*[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25aa\u25b8\*\-]+\s*",
        re.MULTILINE,
    )
    _MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
    _MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean and normalize raw text extracted from documents.

        Steps:
        1. Unicode normalization (NFKD to ASCII compatible).
        2. Replace non-breaking spaces and non-printable control characters.
        3. Remove repetitive divider lines and standalone page numbering.
        4. Standardize bullet points into clean dashes.
        5. Collapse consecutive spaces and empty lines.

        Args:
            text: Raw input string.

        Returns:
            str: Cleaned and normalized text.
        """
        if not text:
            return ""

        # Normalize unicode
        normalized = unicodedata.normalize("NFKD", text)

        # Replace non-breaking spaces and unusual whitespace
        normalized = (
            normalized.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        )

        # Strip unprintable control characters (retain tabs and newlines)
        normalized = "".join(
            char
            for char in normalized
            if char in ("\n", "\t") or (ord(char) >= 32 and ord(char) != 127)
        )

        # Remove standalone page numbers
        normalized = cls._PAGE_NUM_PATTERN.sub("", normalized)

        # Remove decorative line dividers (e.g., ----------- or =============)
        normalized = cls._REPEATED_SYMBOLS_PATTERN.sub("", normalized)

        # Standardize bullets to a simple dash and space
        normalized = cls._BULLET_PATTERN.sub("- ", normalized)

        # Collapse horizontal whitespace
        lines = [
            cls._MULTIPLE_SPACES_PATTERN.sub(" ", line).strip()
            for line in normalized.split("\n")
        ]

        # Filter empty lines and rejoin with single newlines, preserving section breaks
        cleaned_text = "\n".join(lines)
        cleaned_text = cls._MULTIPLE_NEWLINES_PATTERN.sub("\n\n", cleaned_text)

        return cleaned_text.strip()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for LLM budgeting (rule of thumb: ~4 chars per token).

        Args:
            text: Input string.

        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        # Average English word is ~4.5 characters, punctuation adds tokens;
        # len(text) / 4 provides a safe upper-bound heuristic.
        return max(1, len(text) // 4)

    @classmethod
    def optimize_text_for_llm(cls, text: str, max_tokens: int = 3500) -> Tuple[str, bool]:
        """Clean and optionally truncate text to stay within prompt token budgets.

        Args:
            text: Raw input text.
            max_tokens: Maximum allowed estimated tokens.

        Returns:
            Tuple[str, bool]: (Optimized text, boolean indicating whether text was truncated).
        """
        cleaned = cls.clean_text(text)
        current_tokens = cls.estimate_tokens(cleaned)

        if current_tokens <= max_tokens:
            return cleaned, False

        # Budget exceeded: truncate safely at paragraph or sentence boundary
        char_limit = max_tokens * 4
        truncated = cleaned[:char_limit]
        last_break = truncated.rfind("\n\n")
        if last_break > char_limit * 0.8:
            truncated = truncated[:last_break]
        else:
            last_period = truncated.rfind(". ")
            if last_period > char_limit * 0.8:
                truncated = truncated[: last_period + 1]

        truncated = f"{truncated.strip()}\n\n[... Text truncated for token optimization ...]"
        return truncated, True


class SemanticPreMatcher:
    """Computes TF-IDF vector overlap and cosine similarity between JD and Resume."""

    def __init__(self, max_features: int = 1000):
        """Initialize the TF-IDF vectorizer.

        Args:
            max_features: Maximum number of vocabulary features.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=max_features,
            sublinear_tf=True,
        )

    def _extract_top_shared(
        self, tfidf_matrix, feature_names: List[str]
    ) -> List[str]:
        """Extract top shared terms by combined TF-IDF importance."""
        jd_weights = tfidf_matrix[0].toarray()[0]
        res_weights = tfidf_matrix[1].toarray()[0]

        shared_weights = []
        for idx, (jw, rw) in enumerate(zip(jd_weights, res_weights)):
            if jw > 0.0 and rw > 0.0:
                shared_weights.append((feature_names[idx], jw * rw))

        shared_weights.sort(key=lambda item: item[1], reverse=True)
        return [term for term, _ in shared_weights[:10]]

    def analyze(self, jd_text: str, resume_text: str) -> PreMatchResult:
        """Compute lexical/semantic overlap between Job Description and Resume.

        Args:
            jd_text: Cleaned text of the job description.
            resume_text: Cleaned text of the candidate resume.

        Returns:
            PreMatchResult: Object containing similarity score (0-100), shared key terms,
                            and estimated token counts.
        """
        jd_tokens = TextPreprocessor.estimate_tokens(jd_text)
        res_tokens = TextPreprocessor.estimate_tokens(resume_text)

        if not jd_text.strip() or not resume_text.strip():
            return PreMatchResult(
                similarity_score=0.0,
                shared_keywords=[],
                resume_token_estimate=res_tokens,
                jd_token_estimate=jd_tokens,
            )

        try:
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            similarity = float(
                cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            )
            score_percentage = round(min(100.0, max(0.0, similarity * 100.0)), 2)

            feature_names = self.vectorizer.get_feature_names_out().tolist()
            top_shared = self._extract_top_shared(tfidf_matrix, feature_names)

            return PreMatchResult(
                similarity_score=score_percentage,
                shared_keywords=top_shared,
                resume_token_estimate=res_tokens,
                jd_token_estimate=jd_tokens,
            )
        except (ValueError, IndexError, AttributeError):
            return PreMatchResult(
                similarity_score=0.0,
                shared_keywords=[],
                resume_token_estimate=res_tokens,
                jd_token_estimate=jd_tokens,
            )
