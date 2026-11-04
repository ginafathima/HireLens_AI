"""OpenRouter API client with exponential backoff retries and model fallbacks.

Handles communication with OpenRouter's chat completion endpoints,
enforces JSON response formatting, and manages transient 429 rate limits
and server errors gracefully across multiple free-tier models.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple
import requests

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_TIMEOUT,
    get_all_models,
)

logger = logging.getLogger(__name__)


class OpenRouterAPIError(Exception):
    """Base exception for OpenRouter API interactions."""


class AuthenticationError(OpenRouterAPIError):
    """Raised when the OpenRouter API key is missing or invalid."""


class RateLimitExhaustedError(OpenRouterAPIError):
    """Raised when all retry attempts and fallback models hit rate limits."""


class InvalidJSONResponseError(OpenRouterAPIError):
    """Raised when the model fails to return a valid JSON object."""


# pylint: disable=too-few-public-methods
class OpenRouterClient:
    """Client for querying LLMs via OpenRouter with automatic retries and model fallbacks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. Defaults to environment variable.
            base_url: OpenRouter API base URL. Defaults to configuration.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts per model.
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.timeout = timeout or OPENROUTER_TIMEOUT
        self.max_retries = max_retries or OPENROUTER_MAX_RETRIES

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/HireLens_AI",
                "X-Title": "HireLens AI Resume Evaluator",
            }
        )

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code fences and extraneous text from LLM response.

        Args:
            text: Raw string returned by the LLM.

        Returns:
            str: Clean JSON substring.
        """
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return match.group(1).strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1].strip()

        return text

    def _parse_choice_json(self, response: requests.Response, model: str) -> Dict[str, Any]:
        """Validate and parse response payload into a dictionary."""
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterAPIError(f"Model '{model}' returned empty choices array.")

        raw_content = choices[0].get("message", {}).get("content", "")
        clean_content = self._clean_json_response(raw_content)

        try:
            parsed = json.loads(clean_content)
            if not isinstance(parsed, dict):
                raise InvalidJSONResponseError(
                    f"Expected JSON dictionary, got {type(parsed).__name__}."
                )
            parsed["_model_used"] = model
            return parsed
        except json.JSONDecodeError as err:
            raise InvalidJSONResponseError(
                f"Failed to parse LLM output as JSON: {err}"
            ) from err

    def _send_single_request(
        self, endpoint: str, payload: Dict[str, Any], model: str, attempt: int
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Send an HTTP request and process status codes.

        Returns:
            Tuple[Optional[Dict[str, Any]], bool]: (Parsed JSON result, retry_flag).
        """
        try:
            response = self.session.post(endpoint, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                return self._parse_choice_json(response, model), False

            if response.status_code == 401:
                raise AuthenticationError(
                    "Invalid OpenRouter API Key (HTTP 401). Verify OPENROUTER_API_KEY in .env."
                )

            if response.status_code in (429, 500, 502, 503, 504):
                backoff = 2**attempt
                logger.warning(
                    "Model %s returned HTTP %d (attempt %d/%d). Retrying in %ds...",
                    model,
                    response.status_code,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                return None, True

            response.raise_for_status()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_err:
            backoff = 2**attempt
            logger.warning(
                "Network timeout with model %s (attempt %d/%d). Retrying in %ds... Error: %s",
                model,
                attempt,
                self.max_retries,
                backoff,
                conn_err,
            )
            time.sleep(backoff)
            return None, True

        return None, False

    def _query_model_with_retries(
        self, endpoint: str, payload: Dict[str, Any], model: str
    ) -> Optional[Dict[str, Any]]:
        """Attempt to query a single model across its allocated retries."""
        for attempt in range(1, self.max_retries + 1):
            result, should_retry = self._send_single_request(
                endpoint, payload, model, attempt
            )
            if result is not None:
                return result
            if not should_retry:
                break
        return None

    def generate_evaluation(
        self,
        system_prompt: str,
        user_prompt: str,
        preferred_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send evaluation prompts to OpenRouter with retries and fallback models.

        Args:
            system_prompt: System prompt with instructions and schema constraints.
            user_prompt: User prompt containing JD and Resume content.
            preferred_model: Optional override for the primary model.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary response from the LLM.

        Raises:
            AuthenticationError: If the API key is not configured.
            RateLimitExhaustedError: If all models exhaust retry attempts.
            InvalidJSONResponseError: If the model output cannot be parsed as JSON.
        """
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            raise AuthenticationError(
                "Missing OpenRouter API key. Please set OPENROUTER_API_KEY in your .env file."
            )

        model_candidates = [preferred_model] if preferred_model else []
        for m in get_all_models():
            if m not in model_candidates:
                model_candidates.append(m)

        endpoint = f"{self.base_url}/chat/completions"

        for model in model_candidates:
            logger.info("Attempting evaluation with model: %s", model)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            result = self._query_model_with_retries(endpoint, payload, model)
            if result is not None:
                return result

        raise RateLimitExhaustedError(
            "All OpenRouter models failed or rate-limited after maximum retries."
        )
