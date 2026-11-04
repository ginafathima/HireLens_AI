"""Unit tests for text preprocessing, token budgeting, and TF-IDF semantic pre-matching."""

from pathlib import Path
from src.preprocessor import TextPreprocessor, SemanticPreMatcher
from src.parsers import parse_resume

BASE_DIR = Path(__file__).resolve().parent.parent


def test_clean_text_normalization():
    """Verify that multiple spaces, special bullets, and excessive newlines are normalized."""
    dirty_text = (
        "  Candidate Name   \n\n\n\n"
        "• Skill One\n"
        "▪ Skill Two\n"
        "Page 1 of 3\n"
        "------------------\n"
        "Summary:   Experienced in   Python  and    data science.  \n"
    )
    cleaned = TextPreprocessor.clean_text(dirty_text)

    assert "Candidate Name" in cleaned
    assert "Page 1 of 3" not in cleaned
    assert "------------------" not in cleaned
    assert "- Skill One" in cleaned
    assert "- Skill Two" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_estimate_tokens():
    """Verify safe heuristic estimation for token counts."""
    assert TextPreprocessor.estimate_tokens("") == 0
    short_text = "This is a short sample sentence for token testing."
    tokens = TextPreprocessor.estimate_tokens(short_text)
    assert tokens > 0
    assert tokens == max(1, len(short_text) // 4)


def test_optimize_text_for_llm_truncation():
    """Verify token truncation respects maximum boundaries when payload is too large."""
    long_text = "Python data science engineer. " * 500  # ~15,000 characters
    optimized, truncated = TextPreprocessor.optimize_text_for_llm(long_text, max_tokens=200)

    assert truncated is True
    assert TextPreprocessor.estimate_tokens(optimized) <= 250
    assert "[... Text truncated for token optimization ...]" in optimized


def test_semantic_pre_matching_discrimination():
    """Verify TF-IDF gives significantly higher similarity to strong fit than weak fit."""
    jd_file = BASE_DIR / "jd.txt"
    assert jd_file.exists(), "jd.txt benchmark must exist"
    jd_text = TextPreprocessor.clean_text(jd_file.read_text(encoding="utf-8"))

    strong_raw = parse_resume(BASE_DIR / "sample_resumes" / "candidate_strong.pdf")
    weak_raw = parse_resume(BASE_DIR / "sample_resumes" / "candidate_weak.docx")

    strong_clean = TextPreprocessor.clean_text(strong_raw)
    weak_clean = TextPreprocessor.clean_text(weak_raw)

    matcher = SemanticPreMatcher()
    strong_result = matcher.analyze(jd_text, strong_clean)
    weak_result = matcher.analyze(jd_text, weak_clean)

    assert strong_result.similarity_score > weak_result.similarity_score
    # Strong resume has substantial direct keyword overlap
    assert strong_result.similarity_score >= 18.0
    # Strong resume has more than double the overlap of the weak resume
    assert strong_result.similarity_score > weak_result.similarity_score * 2

    # Strong candidate should share key terms like 'python', 'pypdf', etc.
    shared_lower = [k.lower() for k in strong_result.shared_keywords]
    expected_matches = ["python", "data", "pypdf", "fastapi", "rag", "rest"]
    assert any(term in shared_lower for term in expected_matches)
