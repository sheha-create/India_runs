"""
modules/llm_client.py — Thin wrapper around the Groq LLM API.
Uses groq SDK; returns parsed JSON dicts; raises on parse failure.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import config

logger = logging.getLogger(__name__)


def _clean_json(raw: str) -> str:
    """Strip markdown code fences that some models wrap around JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class LLMClient:
    """Unified LLM client using Groq for structured JSON extraction."""

    def __init__(self) -> None:
        self._init_client()

    def _init_client(self) -> None:
        from groq import Groq  # type: ignore
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example → .env and add your key."
            )
        self._client = Groq(api_key=config.GROQ_API_KEY)

    def call(self, prompt: str) -> Dict[str, Any]:
        """Send a prompt and return parsed JSON dict."""
        raw = self._raw_call(prompt)
        try:
            return json.loads(_clean_json(raw))
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned non-JSON. Raw snippet: %s...", raw[:300])
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc

    def _raw_call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""
