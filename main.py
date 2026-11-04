"""HireLens AI - CLI Entrypoint.

Command-line orchestrator for evaluating candidate resumes against
a Job Description using OpenRouter AI models and deterministic weighted scoring.
"""

import argparse
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add project root to sys.path before local imports for clean execution from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from src.config import (
    OPENROUTER_MODEL,
    validate_config,
    ConfigurationError,
)
from src.parsers import (
    parse_resume,
    ParsingError,
)
from src.preprocessor import TextPreprocessor, SemanticPreMatcher, PreMatchResult
from src.openrouter_client import (
    OpenRouterClient,
    OpenRouterAPIError,
    AuthenticationError,
    RateLimitExhaustedError,
)
from src.evaluator import ResumeEvaluator, FinalEvaluationReport


@dataclass
class EvaluationTask:
    """Encapsulates input parameters for executing an evaluation."""

    cleaned_jd: str
    cleaned_resume: str
    pre_summary: str
    pre_score: float
    model: str
    output_path: Path


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HireLens AI - AI-Powered Resume Evaluation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        "-r",
        required=True,
        type=str,
        help="Path to candidate resume file (.pdf or .docx)",
    )
    parser.add_argument(
        "--jd",
        "-j",
        default="jd.txt",
        type=str,
        help="Path to benchmark Job Description file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="evaluation_report.json",
        type=str,
        help="Path to output evaluation report JSON file",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        type=str,
        help="Override OpenRouter model identifier (defaults to .env)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logs",
    )
    return parser.parse_args()


def load_job_description(jd_path: Path) -> str:
    """Load and validate the benchmark Job Description text."""
    if not jd_path.is_file():
        raise FileNotFoundError(
            f"Job Description file not found at: {jd_path.resolve()}\n"
            "Please ensure 'jd.txt' exists in the project root."
        )

    with open(jd_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Job Description file '{jd_path.name}' is empty.")

    return TextPreprocessor.clean_text(content)


def _load_and_preprocess(resume_path: Path, jd_path: Path) -> tuple[str, str]:
    """Ingest and preprocess both JD and Resume documents."""
    logging.info("Reading Job Description from '%s'...", jd_path)
    cleaned_jd = load_job_description(jd_path)
    logging.info("JD loaded (~%d tokens).", TextPreprocessor.estimate_tokens(cleaned_jd))

    logging.info("Extracting resume from '%s'...", resume_path)
    raw_resume = parse_resume(resume_path)
    logging.info("Extracted %d characters from resume.", len(raw_resume))

    cleaned_resume, truncated = TextPreprocessor.optimize_text_for_llm(raw_resume)
    res_tokens = TextPreprocessor.estimate_tokens(cleaned_resume)
    if truncated:
        logging.warning("Resume exceeded token budget; truncated to %d tokens.", res_tokens)
    else:
        logging.info("Resume cleaned (~%d tokens).", res_tokens)

    return cleaned_jd, cleaned_resume


def _execute_ai_pipeline(task: EvaluationTask) -> int:
    """Execute OpenRouter query, validation, and serialization."""
    try:
        logging.info("Connecting to OpenRouter (Model: %s)...", task.model)
        client = OpenRouterClient()

        system_prompt = ResumeEvaluator.build_system_prompt()
        user_prompt = ResumeEvaluator.build_user_prompt(
            task.cleaned_jd, task.cleaned_resume, task.pre_summary
        )

        raw_evaluation = client.generate_evaluation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            preferred_model=task.model,
        )

        model_used = raw_evaluation.get("_model_used", task.model)
        logging.info("Evaluation received from: %s", model_used)

        logging.info("Validating schema and computing deterministic weighted scores...")
        final_report: FinalEvaluationReport = ResumeEvaluator.process_raw_evaluation(
            raw_evaluation
        )

        summary_text = ResumeEvaluator.format_terminal_summary(
            final_report, pre_match_score=task.pre_score
        )
        print(summary_text)

        ResumeEvaluator.save_json_report(final_report, task.output_path)
        logging.info("Evaluation report saved to '%s'.", task.output_path.resolve())
        print(f"[SUCCESS] Report written to: {task.output_path}\n")
        return 0

    except AuthenticationError as auth_err:
        logging.error("Authentication failed: %s", auth_err)
        return 1
    except RateLimitExhaustedError as rle:
        logging.error("Rate limits exhausted across all fallback models: %s", rle)
        return 1
    except OpenRouterAPIError as oae:
        logging.error("OpenRouter API error: %s", oae)
        return 1


def main() -> int:
    """Main CLI execution routine."""
    args = parse_arguments()
    configure_logging(args.verbose)

    print("\n=======================================================")
    print("  HireLens AI - Candidate Resume Evaluation System     ")
    print("=======================================================\n")

    # Step 1 & 2: Ingest and preprocess documents
    try:
        cleaned_jd, cleaned_resume = _load_and_preprocess(
            Path(args.resume), Path(args.jd)
        )
    except (FileNotFoundError, ValueError, ParsingError) as doc_err:
        logging.error("Document ingestion failed: %s", doc_err)
        return 1

    # Step 3: Semantic Pre-Matching
    logging.info("Running TF-IDF lexical overlap analysis...")
    pre_result: PreMatchResult = SemanticPreMatcher().analyze(cleaned_jd, cleaned_resume)
    print(f"\n[Preliminary Analysis] TF-IDF Similarity: {pre_result.similarity_score:.1f}%")
    if pre_result.shared_keywords:
        top_terms = ", ".join(pre_result.shared_keywords[:6])
        print(f"[Preliminary Analysis] Overlapping Key Terms: {top_terms}\n")

    # Step 4: Validate API credentials
    try:
        validate_config(require_api_key=True)
    except ConfigurationError as config_err:
        logging.error(
            "\nConfiguration Error:\n%s\n"
            "To run the AI evaluation, please add your free OpenRouter API key to '.env':\n"
            "OPENROUTER_API_KEY=your_key_here\n"
            "(Get a free key from https://openrouter.ai/keys)",
            config_err,
        )
        return 1

    pre_summary = (
        f"TF-IDF overlap score: {pre_result.similarity_score:.1f}%. "
        f"Shared key terms: {', '.join(pre_result.shared_keywords[:8])}."
    )
    task = EvaluationTask(
        cleaned_jd=cleaned_jd,
        cleaned_resume=cleaned_resume,
        pre_summary=pre_summary,
        pre_score=pre_result.similarity_score,
        model=args.model or OPENROUTER_MODEL,
        output_path=Path(args.output),
    )
    return _execute_ai_pipeline(task)


if __name__ == "__main__":
    sys.exit(main())
