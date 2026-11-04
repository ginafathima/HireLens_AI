"""Evaluation models, Pydantic schemas, and deterministic scoring calculations.

Validates LLM-generated evaluations, enforces schema boundaries,
calculates weighted composite match scores programmatically,
and handles serialization into structured JSON reports and console dashboards.
"""

from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator

from src.config import (
    EVALUATION_WEIGHTS,
    THRESHOLD_MODERATE,
    THRESHOLD_STRONG,
    WEIGHT_DISPLAY,
)


class CategoryScoreDetail(BaseModel):
    """Evaluation score and qualitative rationale for a single dimension."""

    score: int = Field(..., ge=0, le=100, description="Evaluation score from 0 to 100")
    weight: str = Field(..., description="Weight formatted as a percentage string (e.g., '30%')")
    justification: str = Field(..., min_length=5, description="Clear technical rationale")

    @field_validator("score", mode="before")
    @classmethod
    def coerce_and_bound_score(cls, value: Any) -> int:
        """Coerce numeric values and clamp within 0 to 100."""
        try:
            numeric_val = int(round(float(value)))
            return max(0, min(100, numeric_val))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Score must be an integer between 0 and 100, got: {value}") from exc


class CategoryScoresContainer(BaseModel):
    """Container grouping all 5 mandatory evaluation dimensions."""

    core_skills: CategoryScoreDetail
    projects_and_assignments: CategoryScoreDetail
    experience_internships: CategoryScoreDetail
    academic_qualification: CategoryScoreDetail
    resume_quality: CategoryScoreDetail


class RawLLMEvaluation(BaseModel):
    """Pydantic schema to validate the raw unstructured output from the LLM."""

    candidate_name: str = Field(default="Candidate", min_length=1)
    category_scores: CategoryScoresContainer
    key_strengths: List[str] = Field(default_factory=list, min_length=1)
    skill_gaps: List[str] = Field(default_factory=list, min_length=1)
    hiring_manager_recommendation: str = Field(..., min_length=5)


class FinalEvaluationReport(BaseModel):
    """Final, validated report matching the assignment structured output format."""

    candidate_name: str
    overall_match_score: int = Field(..., ge=0, le=100)
    fit_status: str
    category_scores: Dict[str, Dict[str, Any]]
    key_strengths: List[str]
    skill_gaps: List[str]
    hiring_manager_recommendation: str


class ResumeEvaluator:
    """Evaluates candidate fit, validates results, and computes deterministic scores."""

    @staticmethod
    def build_system_prompt() -> str:
        """Construct a rigorous system prompt for the evaluator LLM.

        Returns:
            str: System prompt enforcing technical assessment persona and schema.
        """
        return (
            "You are a Senior Technical Recruiter and Applied AI Hiring Manager conducting a "
            "rigorous, objective candidate evaluation. You evaluate a candidate's resume strictly "
            "against the provided Job Description (JD).\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT fabricate, assume, or hallucinate skills or experience.\n"
            "2. Score each of the 5 categories objectively on a strict 0 to 100 scale:\n"
            "   - core_skills (30% weight): direct overlap with Python, Pandas, NumPy, "
            "pypdf/docx, requests, REST APIs, prompt engineering, embeddings.\n"
            "   - projects_and_assignments (25% weight): depth, deployment evidence, complexity.\n"
            "   - experience_internships (15% weight): production engineering, internships, "
            "real-world bug solving, open-source.\n"
            "   - academic_qualification (15% weight): CS/AI/Math degree, 2024-2026 graduation.\n"
            "   - resume_quality (15% weight): structure, quantifiable metrics, no buzzwords.\n"
            "3. Provide concise, high-signal technical justifications for each score.\n"
            "4. Return ONLY a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "candidate_name": "Extracted or inferred full name",\n'
            '  "category_scores": {\n'
            '    "core_skills": {"score": 85, "justification": "..."},\n'
            '    "projects_and_assignments": {"score": 80, "justification": "..."},\n'
            '    "experience_internships": {"score": 75, "justification": "..."},\n'
            '    "academic_qualification": {"score": 90, "justification": "..."},\n'
            '    "resume_quality": {"score": 85, "justification": "..."}\n'
            "  },\n"
            '  "key_strengths": ["Strength 1", "Strength 2"],\n'
            '  "skill_gaps": ["Gap 1", "Gap 2"],\n'
            '  "hiring_manager_recommendation": "Advance to technical interview round..."\n'
            "}"
        )

    @staticmethod
    def build_user_prompt(jd_text: str, resume_text: str, pre_match_summary: str = "") -> str:
        """Construct the evaluation prompt containing the JD and candidate resume.

        Args:
            jd_text: Cleaned text of the job description.
            resume_text: Cleaned text of the resume.
            pre_match_summary: Optional lexical pre-match context.

        Returns:
            str: User prompt string.
        """
        prompt = (
            f"=== TARGET JOB DESCRIPTION ===\n{jd_text}\n\n"
            f"=== CANDIDATE RESUME ===\n{resume_text}\n\n"
        )
        if pre_match_summary:
            prompt += f"=== PRELIMINARY TF-IDF OVERLAP ANALYSIS ===\n{pre_match_summary}\n\n"

        prompt += (
            "Evaluate the candidate against the Job Description and return the "
            "structured JSON evaluation."
        )
        return prompt

    @staticmethod
    def calculate_deterministic_score(categories: CategoryScoresContainer) -> int:
        """Calculate final overall match score programmatically using mandatory weights.

        Formula:
            Overall Score = round(
                core_skills * 0.30 +
                projects_and_assignments * 0.25 +
                experience_internships * 0.15 +
                academic_qualification * 0.15 +
                resume_quality * 0.15
            )

        Args:
            categories: Validated CategoryScoresContainer.

        Returns:
            int: Deterministic score between 0 and 100.
        """
        weights = EVALUATION_WEIGHTS
        raw_score = (
            categories.core_skills.score * weights["core_skills"]
            + categories.projects_and_assignments.score * weights["projects_and_assignments"]
            + categories.experience_internships.score * weights["experience_internships"]
            + categories.academic_qualification.score * weights["academic_qualification"]
            + categories.resume_quality.score * weights["resume_quality"]
        )
        return int(round(raw_score))

    @staticmethod
    def determine_fit_status(overall_score: int) -> str:
        """Classify fit status based on overall match score.

        Args:
            overall_score: Score between 0 and 100.

        Returns:
            str: 'Strong Match' (>=80), 'Moderate Match' (60-79), or 'Weak Match' (<60).
        """
        if overall_score >= THRESHOLD_STRONG:
            return "Strong Match"
        if overall_score >= THRESHOLD_MODERATE:
            return "Moderate Match"
        return "Weak Match"

    @classmethod
    def process_raw_evaluation(cls, raw_data: Dict[str, Any]) -> FinalEvaluationReport:
        """Validate raw LLM dictionary and build the programmatic final report.

        Args:
            raw_data: Dictionary returned from OpenRouterClient.

        Returns:
            FinalEvaluationReport: Validated and deterministically scored report.
        """
        # Ensure default weights are injected into category_scores if absent from LLM
        cat_scores = raw_data.get("category_scores", {})
        if isinstance(cat_scores, dict):
            for key, weight_str in WEIGHT_DISPLAY.items():
                if key in cat_scores and isinstance(cat_scores[key], dict):
                    cat_scores[key]["weight"] = weight_str

        # Validate with Pydantic
        validated_raw = RawLLMEvaluation.model_validate(raw_data)

        # Deterministic score calculation (Rule 8: calculate programmatically, not via AI)
        overall_score = cls.calculate_deterministic_score(validated_raw.category_scores)
        fit_status = cls.determine_fit_status(overall_score)

        # Build output structure matching assignment Section 3.4
        formatted_categories = {
            "core_skills": {
                "score": validated_raw.category_scores.core_skills.score,
                "weight": WEIGHT_DISPLAY["core_skills"],
                "justification": validated_raw.category_scores.core_skills.justification,
            },
            "projects_and_assignments": {
                "score": validated_raw.category_scores.projects_and_assignments.score,
                "weight": WEIGHT_DISPLAY["projects_and_assignments"],
                "justification": (
                    validated_raw.category_scores.projects_and_assignments.justification
                ),
            },
            "experience_internships": {
                "score": validated_raw.category_scores.experience_internships.score,
                "weight": WEIGHT_DISPLAY["experience_internships"],
                "justification": validated_raw.category_scores.experience_internships.justification,
            },
            "academic_qualification": {
                "score": validated_raw.category_scores.academic_qualification.score,
                "weight": WEIGHT_DISPLAY["academic_qualification"],
                "justification": validated_raw.category_scores.academic_qualification.justification,
            },
            "resume_quality": {
                "score": validated_raw.category_scores.resume_quality.score,
                "weight": WEIGHT_DISPLAY["resume_quality"],
                "justification": validated_raw.category_scores.resume_quality.justification,
            },
        }

        return FinalEvaluationReport(
            candidate_name=validated_raw.candidate_name,
            overall_match_score=overall_score,
            fit_status=fit_status,
            category_scores=formatted_categories,
            key_strengths=validated_raw.key_strengths,
            skill_gaps=validated_raw.skill_gaps,
            hiring_manager_recommendation=validated_raw.hiring_manager_recommendation,
        )

    @staticmethod
    def save_json_report(report: FinalEvaluationReport, output_path: Path) -> None:
        """Serialize evaluation report to disk as formatted JSON.

        Args:
            report: FinalEvaluationReport instance.
            output_path: Target file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    @staticmethod
    def format_terminal_summary(report: FinalEvaluationReport, pre_match_score: float = 0.0) -> str:
        """Render a clean ASCII dashboard of evaluation results for terminal output.

        Args:
            report: FinalEvaluationReport instance.
            pre_match_score: Optional TF-IDF preliminary score.

        Returns:
            str: Multi-line formatted terminal string.
        """
        divider = "=" * 76
        sub_divider = "-" * 76

        lines = [
            "",
            divider,
            "  HIRELENS AI - CANDIDATE EVALUATION DASHBOARD",
            divider,
            f"  Candidate Name:       {report.candidate_name}",
            f"  Overall Match Score:  {report.overall_match_score}/100",
            f"  Fit Status:           {report.fit_status}",
            f"  TF-IDF Pre-Score:     {pre_match_score:.1f}% lexical overlap",
            sub_divider,
            f"  {'CATEGORY':<30} {'WEIGHT':<10} {'SCORE':<10} {'STATUS'}",
            sub_divider,
        ]

        for key, details in report.category_scores.items():
            label = key.replace("_", " ").title()
            score = details["score"]
            weight = details["weight"]
            badge = "[EXCELLENT]" if score >= 80 else ("[GOOD]" if score >= 60 else "[NEEDS WORK]")
            lines.append(f"  {label:<30} {weight:<10} {score:<10} {badge}")

        lines.extend([
            sub_divider,
            "  KEY STRENGTHS:",
        ])
        for s in report.key_strengths:
            lines.append(f"   + {s}")

        lines.extend([
            "",
            "  SKILL GAPS & AREAS FOR IMPROVEMENT:",
        ])
        for g in report.skill_gaps:
            lines.append(f"   - {g}")

        lines.extend([
            "",
            "  HIRING MANAGER RECOMMENDATION:",
            f"   >> {report.hiring_manager_recommendation}",
            divider,
            "",
        ])

        return "\n".join(lines)
