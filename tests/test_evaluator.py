"""Unit tests for deterministic scoring, Pydantic schemas, and report formatting."""

# pylint: disable=redefined-outer-name

import json
import pytest

from src.evaluator import (
    CategoryScoreDetail,
    CategoryScoresContainer,
    FinalEvaluationReport,
    ResumeEvaluator,
)


@pytest.fixture
def sample_category_container():
    """Build a sample valid CategoryScoresContainer."""
    return CategoryScoresContainer(
        core_skills=CategoryScoreDetail(
            score=85, weight="30%", justification="Solid Python and pypdf."
        ),
        projects_and_assignments=CategoryScoreDetail(
            score=80, weight="25%", justification="Good end-to-end projects."
        ),
        experience_internships=CategoryScoreDetail(
            score=75, weight="15%", justification="Relevant 6-month internship."
        ),
        academic_qualification=CategoryScoreDetail(
            score=90, weight="15%", justification="B.Tech CS 2025 batch."
        ),
        resume_quality=CategoryScoreDetail(
            score=85, weight="15%", justification="Clear metrics and structure."
        ),
    )


def test_deterministic_score_calculation(sample_category_container):
    """Verify that Python calculates the exact weighted overall score programmatically.

    Calculation:
    (85 * 0.30) + (80 * 0.25) + (75 * 0.15) + (90 * 0.15) + (85 * 0.15)
    = 25.5 + 20.0 + 11.25 + 13.5 + 12.75 = 83.0 -> 83
    """
    calculated_score = ResumeEvaluator.calculate_deterministic_score(
        sample_category_container
    )
    assert calculated_score == 83


def test_fit_status_thresholds():
    """Verify fit status classifications."""
    assert ResumeEvaluator.determine_fit_status(95) == "Strong Match"
    assert ResumeEvaluator.determine_fit_status(80) == "Strong Match"
    assert ResumeEvaluator.determine_fit_status(79) == "Moderate Match"
    assert ResumeEvaluator.determine_fit_status(60) == "Moderate Match"
    assert ResumeEvaluator.determine_fit_status(59) == "Weak Match"
    assert ResumeEvaluator.determine_fit_status(30) == "Weak Match"


def test_category_score_clamping():
    """Verify that scores are safely coerced and clamped between 0 and 100."""
    score_high = CategoryScoreDetail(
        score=120, weight="30%", justification="Exceeds bounds"
    )
    assert score_high.score == 100

    score_low = CategoryScoreDetail(
        score=-10, weight="30%", justification="Sub-zero bound"
    )
    assert score_low.score == 0

    score_float = CategoryScoreDetail(
        score=78.6, weight="25%", justification="Float coercion"
    )
    assert score_float.score == 79


def test_process_raw_evaluation_flow():
    """Verify end-to-end transformation from raw LLM dictionary to FinalEvaluationReport."""
    raw_llm_payload = {
        "candidate_name": "Aarav Sharma",
        "category_scores": {
            "core_skills": {
                "score": 90,
                "justification": "Mastery of Python, pypdf, and FastAPI.",
            },
            "projects_and_assignments": {
                "score": 85,
                "justification": "DocuIntellect project shows applied ML.",
            },
            "experience_internships": {
                "score": 80,
                "justification": "6 months AI engineering internship.",
            },
            "academic_qualification": {
                "score": 95,
                "justification": "B.Tech CS 2025 batch from KTU.",
            },
            "resume_quality": {
                "score": 85,
                "justification": "Metrics included; clean formatting.",
            },
        },
        "key_strengths": ["Strong Python proficiency", "RAG and Document AI experience"],
        "skill_gaps": ["No production experience with Ollama or local LLM runtimes"],
        "hiring_manager_recommendation": (
            "Strong candidate. Recommend advancing to technical interview."
        ),
    }

    report = ResumeEvaluator.process_raw_evaluation(raw_llm_payload)
    assert isinstance(report, FinalEvaluationReport)
    assert report.candidate_name == "Aarav Sharma"
    # (90*0.30) + (85*0.25) + (80*0.15) + (95*0.15) + (85*0.15)
    # = 27.0 + 21.25 + 12.0 + 14.25 + 12.75 = 87.25 -> 87
    assert report.overall_match_score == 87
    assert report.fit_status == "Strong Match"
    assert report.category_scores["core_skills"]["weight"] == "30%"


def test_save_and_read_json_report(tmp_path):
    """Verify JSON file persistence and schema consistency."""
    report = FinalEvaluationReport(
        candidate_name="Test Candidate",
        overall_match_score=82,
        fit_status="Strong Match",
        category_scores={
            "core_skills": {
                "score": 85,
                "weight": "30%",
                "justification": "Good Python.",
            },
            "projects_and_assignments": {
                "score": 80,
                "weight": "25%",
                "justification": "Nice projects.",
            },
            "experience_internships": {
                "score": 75,
                "weight": "15%",
                "justification": "Good internship.",
            },
            "academic_qualification": {
                "score": 90,
                "weight": "15%",
                "justification": "CS degree.",
            },
            "resume_quality": {
                "score": 85,
                "weight": "15%",
                "justification": "Clean format.",
            },
        },
        key_strengths=["Python", "FastAPI"],
        skill_gaps=["No Docker"],
        hiring_manager_recommendation="Proceed to interview.",
    )

    out_file = tmp_path / "evaluation_report.json"
    ResumeEvaluator.save_json_report(report, out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data["candidate_name"] == "Test Candidate"
    assert loaded_data["overall_match_score"] == 82
    assert loaded_data["fit_status"] == "Strong Match"
