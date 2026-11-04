"""Configuration module for the HireLens AI application.

Loads environment variables from .env, validates critical settings,
and defines immutable evaluation weights and thresholds.
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env if present
load_dotenv(dotenv_path=ENV_PATH)


class ConfigurationError(Exception):
    """Raised when application configuration is missing or invalid."""


# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-001"
).strip()

_FALLBACK_DEFAULT = (
    "meta-llama/llama-3.3-70b-instruct:free,"
    "google/gemma-2-9b-it:free,"
    "mistralai/mistral-7b-instruct:free"
)
_raw_fallbacks = os.getenv("OPENROUTER_FALLBACK_MODELS", _FALLBACK_DEFAULT)
OPENROUTER_FALLBACK_MODELS: List[str] = [
    model.strip() for model in _raw_fallbacks.split(",") if model.strip()
]

# Network settings
try:
    OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "45"))
except ValueError:
    OPENROUTER_TIMEOUT = 45

try:
    OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
except ValueError:
    OPENROUTER_MAX_RETRIES = 3

# Mandatory evaluation weights (must sum to 1.00)
EVALUATION_WEIGHTS: Dict[str, float] = {
    "core_skills": 0.30,
    "projects_and_assignments": 0.25,
    "experience_internships": 0.15,
    "academic_qualification": 0.15,
    "resume_quality": 0.15,
}

# Display string weights for final report
WEIGHT_DISPLAY: Dict[str, str] = {
    "core_skills": "30%",
    "projects_and_assignments": "25%",
    "experience_internships": "15%",
    "academic_qualification": "15%",
    "resume_quality": "15%",
}

# Score thresholds for fit status
THRESHOLD_STRONG = 80
THRESHOLD_MODERATE = 60


def get_all_models() -> List[str]:
    """Return an ordered list of models: primary model followed by unique fallback models.

    Returns:
        List[str]: Ordered list of model identifiers to attempt.
    """
    models = [OPENROUTER_MODEL]
    for fallback in OPENROUTER_FALLBACK_MODELS:
        if fallback not in models:
            models.append(fallback)
    return models


def validate_config(require_api_key: bool = True) -> None:
    """Validate runtime configuration settings.

    Args:
        require_api_key: Whether to enforce presence of OPENROUTER_API_KEY.

    Raises:
        ConfigurationError: If any configuration value is invalid.
    """
    if require_api_key:
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not set or contains the default placeholder. "
                "Please configure a valid OpenRouter API key in your .env file."
            )

    weight_sum = round(sum(EVALUATION_WEIGHTS.values()), 4)
    if weight_sum != 1.0:
        raise ConfigurationError(
            f"Evaluation weights must sum to 1.00 (got {weight_sum})."
        )
