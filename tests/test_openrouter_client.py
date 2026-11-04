"""Unit tests for OpenRouter client, JSON extraction, and fallback logic."""

# pylint: disable=protected-access

from unittest.mock import patch, MagicMock
import pytest

from src.openrouter_client import (
    OpenRouterClient,
    AuthenticationError,
)


def test_clean_json_response_markdown_fences():
    """Verify markdown fences and whitespace are stripped to isolate raw JSON."""
    client = OpenRouterClient(api_key="test_key")

    wrapped = '```json\n{"candidate_name": "Jane Doe", "score": 85}\n```'
    clean = client._clean_json_response(wrapped)
    assert clean == '{"candidate_name": "Jane Doe", "score": 85}'

    wrapped_plain = '```\n{"candidate_name": "Jane Doe"}\n```'
    clean_plain = client._clean_json_response(wrapped_plain)
    assert clean_plain == '{"candidate_name": "Jane Doe"}'

    surrounded = 'Here is your evaluation:\n{"candidate_name": "Jane Doe"}\nThank you!'
    clean_surrounded = client._clean_json_response(surrounded)
    assert clean_surrounded == '{"candidate_name": "Jane Doe"}'


def test_missing_api_key_raises_auth_error():
    """Verify that calling generate_evaluation without an API key raises AuthenticationError."""
    client = OpenRouterClient(api_key="")
    with pytest.raises(AuthenticationError):
        client.generate_evaluation("system prompt", "user prompt")


@patch("src.openrouter_client.requests.Session.post")
def test_successful_api_response(mock_post):
    """Verify successful 200 JSON parsing from OpenRouter."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"candidate_name": "Test User", "category_scores": {}, '
                        '"key_strengths": ["Python"], "skill_gaps": [], '
                        '"hiring_manager_recommendation": "Advance"}'
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="sk-or-v1-validkey")
    result = client.generate_evaluation("system", "user")

    assert result["candidate_name"] == "Test User"
    assert "_model_used" in result


@patch("src.openrouter_client.requests.Session.post")
def test_retry_on_429_then_success(mock_post):
    """Verify exponential retry handles a transient 429 rate limit."""
    resp_429 = MagicMock()
    resp_429.status_code = 429

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"candidate_name": "Recovered User", "category_scores": {}, '
                        '"key_strengths": [], "skill_gaps": [], '
                        '"hiring_manager_recommendation": "Advance"}'
                    )
                }
            }
        ]
    }

    mock_post.side_effect = [resp_429, resp_200]

    with patch("src.openrouter_client.time.sleep", return_value=None):
        client = OpenRouterClient(api_key="sk-or-v1-validkey", max_retries=2)
        result = client.generate_evaluation("system", "user")

    assert result["candidate_name"] == "Recovered User"
    assert mock_post.call_count == 2
