"""
modules/candidate_profiler.py — Candidate Profiling Module (PRD §7.2)

Reads raw candidate records (from a pandas DataFrame row / dict) and
normalizes them into CandidateProfile objects using LLM extraction on
free-text fields.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from modules.llm_client import LLMClient
from schemas.candidate_profile import CandidateProfile, WorkExperience

logger = logging.getLogger(__name__)

_PROFILE_PROMPT = """\
You are an expert resume analyst and technical recruiter.

Analyze the candidate record below and extract a normalized profile.
Respond ONLY with valid JSON — no markdown, no explanation.

JSON schema:
{{
  "explicit_skills": ["list of skills explicitly stated in the record"],
  "inferred_skills": ["skills INFERRED from project/work descriptions, not explicitly listed"],
  "total_years_experience": float,
  "career_trajectory": "2-sentence description of the candidate's career progression pattern (e.g. IC → tech lead, domain specialist, breadth generalist)",
  "domain_exposure": ["list of industries/domains they've worked in"],
  "behavioral_signals": ["behavioral indicators from the text: leadership examples, collaboration, ownership, delivery under pressure, etc."],
  "platform_signals": ["GitHub/LinkedIn/portfolio signals: open source activity, publications, community involvement, etc."],
  "work_history": [
    {{
      "company": "string",
      "title": "string",
      "duration_years": float or null,
      "description": "brief description of role",
      "domain": "industry/domain of this role"
    }}
  ]
}}

Candidate Record:
---
Name: {name}
Current Title: {current_title}
Years Experience: {years_experience}
Skills (listed): {skills}
Education: {education}
Summary: {summary}
Work History: {work_history}
Projects: {projects}
Behavioral Notes: {behavioral_notes}
GitHub URL: {github_url}
LinkedIn URL: {linkedin_url}
---
"""


def _safe_str(val: Any, default: str = "") -> str:
    """Convert a value to string, handling NaN/None gracefully."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _infer_platform_signals(row: Dict[str, Any]) -> List[str]:
    """Extract platform signals directly from structured fields without LLM."""
    signals = []
    github = _safe_str(row.get("github_url"))
    linkedin = _safe_str(row.get("linkedin_url"))
    if github and github.lower() not in ("", "n/a", "none"):
        signals.append(f"GitHub profile: {github}")
    if linkedin and linkedin.lower() not in ("", "n/a", "none"):
        signals.append(f"LinkedIn profile: {linkedin}")
    return signals


def map_jsonl_to_flat_dict(cand: dict) -> dict:
    """Map raw nested candidate dictionary (from candidates.jsonl) to the flat format expected by CandidateProfiler."""
    prof = cand.get("profile", {})
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    edu = cand.get("education", [])
    signals = cand.get("redrob_signals", {})
    
    # Skills raw
    skills_raw = ", ".join(s.get("name", "") for s in skills if s.get("name"))
    
    # Work history raw
    work_history_raw = "; ".join(
        f"{job.get('title', '')} at {job.get('company', '')} ({job.get('duration_months', 0)}mo): {job.get('description', '')}"
        for job in career
    )
    
    # Education raw
    edu_raw = "; ".join(
        f"{e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution', '')} ({e.get('tier', '')})"
        for e in edu
    )
    
    return {
        "candidate_id": cand.get("candidate_id", ""),
        "name": prof.get("anonymized_name", "Unknown"),
        "email": "",
        "current_title": prof.get("current_title", ""),
        "years_experience": float(prof.get("years_of_experience", 0.0) or 0.0),
        "skills": skills_raw,
        "summary": prof.get("summary", ""),
        "work_history": work_history_raw,
        "projects": "",
        "behavioral_notes": "",
        "github_url": f"https://github.com/activity-{signals.get('github_activity_score')}" if signals.get("github_activity_score", -1) >= 0 else "",
        "linkedin_url": "linked-in" if signals.get("linkedin_connected") else "",
        "education": edu_raw,
    }


class CandidateProfiler:
    """Converts raw candidate data rows into normalized CandidateProfile objects."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_llm: bool = True,
    ) -> None:
        self._use_llm = use_llm
        if use_llm:
            self._llm = llm_client or LLMClient()
        else:
            self._llm = None

    def profile_row(self, row: Dict[str, Any]) -> CandidateProfile:
        """
        Build a CandidateProfile from a single candidate record dict.
        Falls back to structured-only parsing if LLM call fails.
        """
        candidate_id = _safe_str(row.get("candidate_id", row.get("id", "")))
        name = _safe_str(row.get("name", "Unknown"))
        email = _safe_str(row.get("email", ""))
        current_title = _safe_str(row.get("current_title", row.get("title", "")))
        years_experience = _safe_float(row.get("years_experience", row.get("experience_years", 0)))
        skills_raw = _safe_str(row.get("skills", ""))
        summary = _safe_str(row.get("summary", ""))
        work_history_raw = _safe_str(row.get("work_history", ""))
        projects = _safe_str(row.get("projects", ""))
        behavioral_notes = _safe_str(row.get("behavioral_notes", ""))
        github_url = _safe_str(row.get("github_url", ""))
        linkedin_url = _safe_str(row.get("linkedin_url", ""))
        education = _safe_str(row.get("education", ""))

        # Build explicit skills from structured field
        explicit_skills = [s.strip() for s in re.split(r"[,;|]", skills_raw) if s.strip()]

        # Platform signals from structured fields
        platform_signals = _infer_platform_signals(row)

        llm_data = {}
        if self._use_llm and self._llm is not None:
            llm_data = self._call_llm(
                name=name,
                current_title=current_title,
                years_experience=years_experience,
                skills=skills_raw,
                education=education,
                summary=summary,
                work_history=work_history_raw,
                projects=projects,
                behavioral_notes=behavioral_notes,
                github_url=github_url,
                linkedin_url=linkedin_url,
            )

        # Merge LLM output with structured fields
        inferred_skills = llm_data.get("inferred_skills", [])
        career_trajectory = llm_data.get("career_trajectory", "")
        domain_exposure = llm_data.get("domain_exposure", [])
        behavioral_signals = llm_data.get("behavioral_signals", [])
        platform_signals += [
            s for s in llm_data.get("platform_signals", [])
            if s not in platform_signals
        ]

        # Parse work history from LLM output
        work_history_objs: List[WorkExperience] = []
        for wh in llm_data.get("work_history", []):
            if isinstance(wh, dict):
                work_history_objs.append(WorkExperience(**{
                    k: v for k, v in wh.items()
                    if k in WorkExperience.model_fields
                }))

        total_years = llm_data.get("total_years_experience", years_experience) or years_experience

        return CandidateProfile(
            candidate_id=candidate_id or name.replace(" ", "_").lower(),
            name=name,
            email=email,
            current_title=current_title,
            explicit_skills=explicit_skills,
            inferred_skills=inferred_skills,
            total_years_experience=float(total_years),
            work_history=work_history_objs,
            career_trajectory=career_trajectory,
            domain_exposure=domain_exposure,
            behavioral_signals=behavioral_signals,
            platform_signals=platform_signals,
            raw_summary=summary,
            raw_projects=projects,
            education=education,
        )

    def _call_llm(self, **kwargs: Any) -> dict:
        """Call LLM for profile extraction; returns empty dict on failure."""
        if self._llm is None:
            return {}
        prompt = _PROFILE_PROMPT.format(**{k: v or "N/A" for k, v in kwargs.items()})
        try:
            return self._llm.call(prompt)
        except Exception as exc:
            logger.warning("LLM profiling failed for %r: %s", kwargs.get("name"), exc)
            return {}

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        show_progress: bool = True,
    ) -> List[CandidateProfile]:
        """
        Build profiles for all rows in a DataFrame.

        Args:
            df: Candidate dataset DataFrame.
            show_progress: Show tqdm progress bar.
        Returns:
            List of CandidateProfile objects.
        """
        profiles: List[CandidateProfile] = []
        rows = df.to_dict(orient="records")
        iterator = tqdm(rows, desc="Profiling candidates", unit="candidate") if show_progress else rows
        for row in iterator:
            try:
                profile = self.profile_row(row)
                profiles.append(profile)
            except Exception as exc:
                logger.error("Failed to profile candidate %r: %s", row.get("name"), exc)
        return profiles
