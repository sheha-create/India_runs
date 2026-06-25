"""
modules/jd_parser.py — JD Understanding Module (PRD §7.1)

Sends raw JD text to the LLM with a structured extraction prompt and
returns a validated RoleProfile Pydantic object.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from modules.llm_client import LLMClient
from schemas.role_profile import RoleProfile

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are an expert technical recruiter and job description analyst.

Analyze the job description below and extract a structured role profile.
Respond ONLY with a valid JSON object — no markdown, no extra text.

JSON schema to follow:
{{
  "role_title": "string — the job title",
  "seniority_level": "string — one of: Intern, Junior, Mid-level, Senior, Staff, Principal, Lead, Director, VP",
  "required_skills": ["list of must-have technical skills, tools, languages, frameworks"],
  "preferred_skills": ["list of nice-to-have skills"],
  "domain": "string — industry or domain (e.g. FinTech, Healthcare, SaaS, E-commerce)",
  "years_experience_min": integer or null,
  "behavioral_expectations": ["soft-skill or cultural signals inferred from JD wording (e.g. 'self-starter', 'thrives in fast-paced environment', 'cross-functional collaborator')"],
  "implicit_signals": ["unstated but implied expectations (e.g. 'startup background preferred', 'open-source contributions valued', 'expectation of on-call rotation')"],
  "responsibilities_summary": "2–3 sentence prose summary of what this person will actually do day-to-day"
}}

Job Description:
---
{jd_text}
---
"""


class JDParser:
    """Parses raw job description text into a structured RoleProfile."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        try:
            self._llm = llm_client or LLMClient()
        except Exception as exc:
            logger.warning("LLM client initialization failed: %s", exc)
            self._llm = None

    def parse(self, jd_text: str) -> RoleProfile:
        """
        Args:
            jd_text: Raw job description text (any format).
        Returns:
            RoleProfile: Validated structured role profile.
        """
        if not jd_text or not jd_text.strip():
            raise ValueError("Job description text is empty.")

        if self._llm is None:
            logger.warning("LLM not available, returning minimal profile")
            return RoleProfile(
                role_title="Unknown Role",
                seniority_level="Unknown",
            )

        prompt = _EXTRACTION_PROMPT.format(jd_text=jd_text.strip())
        logger.info("Sending JD to LLM for structured extraction...")

        try:
            data = self._llm.call(prompt)
        except ValueError as exc:
            logger.error("JD parsing failed: %s", exc)
            # Return a minimal profile so the pipeline can continue
            return RoleProfile(
                role_title="Unknown Role",
                seniority_level="Unknown",
            )

        # Pydantic will coerce / validate; unknown extra fields are ignored
        try:
            profile = RoleProfile(**data)
        except Exception as exc:  # pragma: no cover
            logger.warning("Pydantic validation issues, attempting partial parse: %s", exc)
            # Filter to only known fields and drop None values so defaults apply
            known = RoleProfile.model_fields.keys()
            filtered = {k: v for k, v in data.items() if k in known and v is not None}
            profile = RoleProfile(**filtered)

        logger.info(
            "JD parsed -> role=%r seniority=%r required_skills=%d",
            profile.role_title,
            profile.seniority_level,
            len(profile.required_skills),
        )
        return profile
