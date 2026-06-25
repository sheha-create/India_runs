"""
modules/ranker.py — Matching/Ranking Engine (PRD §7.3)

Hybrid two-stage ranking:
  1. Embedding similarity pre-filter (done in embedder.py → top-N passed here)
  2. LLM reasoning pass for nuanced fit judgment + rationale per candidate
  3. Weighted aggregation into final score
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from tqdm import tqdm

import config
from modules.llm_client import LLMClient
from schemas.candidate_profile import CandidateProfile, ScoredCandidate
from schemas.role_profile import RoleProfile

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a senior technical recruiter performing a structured candidate evaluation.

Below is a job role profile followed by a candidate profile.
Your task: assess how well this candidate fits the role across 4 dimensions.

Respond ONLY with valid JSON — no markdown, no explanation.

JSON schema:
{{
  "skill_match": integer 0-100,
  "experience_relevance": integer 0-100,
  "behavioral_fit": integer 0-100,
  "platform_signal": integer 0-100,
  "rationale": "3-4 sentence explanation of the overall fit assessment, citing specific evidence from the candidate profile",
  "skill_evidence": ["specific skills the candidate has that match role requirements"],
  "experience_evidence": ["specific experiences/roles that are relevant to this position"]
}}

Scoring guide:
- skill_match: how well candidate's explicit + inferred skills match required and preferred skills
- experience_relevance: how relevant the candidate's career trajectory, domain, seniority, and tenure are
- behavioral_fit: how well behavioral signals match the role's cultural/soft expectations
- platform_signal: value from GitHub/LinkedIn/platform activity (if none visible, score 50 as neutral, not 0)

---
ROLE PROFILE:
Title: {role_title}
Seniority: {seniority_level}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Domain: {domain}
Behavioral Expectations: {behavioral_expectations}
Responsibilities: {responsibilities_summary}

---
CANDIDATE PROFILE:
Name: {name}
Current Title: {current_title}
Years Experience: {total_years_experience}
Skills: {all_skills}
Career Trajectory: {career_trajectory}
Domain Exposure: {domain_exposure}
Behavioral Signals: {behavioral_signals}
Platform Signals: {platform_signals}
Education: {education}
Summary: {raw_summary}
---
"""


def _build_judge_prompt(role: RoleProfile, candidate: CandidateProfile) -> str:
    all_skills = list(set(candidate.explicit_skills + candidate.inferred_skills))
    return _JUDGE_PROMPT.format(
        role_title=role.role_title,
        seniority_level=role.seniority_level,
        required_skills=", ".join(role.required_skills),
        preferred_skills=", ".join(role.preferred_skills),
        domain=role.domain,
        behavioral_expectations=", ".join(role.behavioral_expectations),
        responsibilities_summary=role.responsibilities_summary,
        name=candidate.name,
        current_title=candidate.current_title,
        total_years_experience=candidate.total_years_experience,
        all_skills=", ".join(all_skills),
        career_trajectory=candidate.career_trajectory,
        domain_exposure=", ".join(candidate.domain_exposure),
        behavioral_signals=", ".join(candidate.behavioral_signals),
        platform_signals=", ".join(candidate.platform_signals),
        education=candidate.education,
        raw_summary=candidate.raw_summary[:500] if candidate.raw_summary else "",
    )


def _compute_final_score(
    skill_match: float,
    experience_relevance: float,
    behavioral_fit: float,
    platform_signal: float,
    embedding_similarity: float,
) -> float:
    w = config.WEIGHTS
    score = (
        skill_match * w["skill_match"]
        + experience_relevance * w["experience_relevance"]
        + behavioral_fit * w["behavioral_fit"]
        + platform_signal * w["platform_signal"]
        + embedding_similarity * w["embedding_similarity"]
    )
    return round(min(100.0, max(0.0, score)), 2)


class Ranker:
    """LLM-powered candidate ranking engine with explainable scores."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        try:
            self._llm = llm_client or LLMClient()
        except Exception as exc:
            logger.warning("LLM client initialization failed: %s", exc)
            self._llm = None

    def rank(
        self,
        role_profile: RoleProfile,
        candidates_with_scores: List[Tuple[CandidateProfile, float]],
        show_progress: bool = True,
    ) -> List[ScoredCandidate]:
        """
        Run the LLM reasoning pass on pre-filtered (embedding-ranked) candidates.

        Args:
            role_profile: Structured role requirements.
            candidates_with_scores: List of (profile, embedding_similarity_score) from Embedder.
            show_progress: Show tqdm progress bar.

        Returns:
            List of ScoredCandidate sorted by overall_score descending, with rank assigned.
        """
        scored: List[ScoredCandidate] = []
        iterator = (
            tqdm(candidates_with_scores, desc="LLM reasoning pass", unit="candidate")
            if show_progress
            else candidates_with_scores
        )

        for candidate, embedding_score in iterator:
            if self._llm is not None:
                llm_scores = self._judge_candidate(role_profile, candidate)
            else:
                llm_scores = {
                    "skill_match": 50,
                    "experience_relevance": 50,
                    "behavioral_fit": 50,
                    "platform_signal": 50,
                    "rationale": f"LLM not available. Embedding similarity: {embedding_score:.1f}.",
                    "skill_evidence": [],
                    "experience_evidence": [],
                }

            skill_match = float(llm_scores.get("skill_match", 50))
            experience_relevance = float(llm_scores.get("experience_relevance", 50))
            behavioral_fit = float(llm_scores.get("behavioral_fit", 50))
            platform_signal = float(llm_scores.get("platform_signal", 50))

            overall = _compute_final_score(
                skill_match=skill_match,
                experience_relevance=experience_relevance,
                behavioral_fit=behavioral_fit,
                platform_signal=platform_signal,
                embedding_similarity=embedding_score,
            )

            scored.append(
                ScoredCandidate(
                    profile=candidate,
                    skill_match=skill_match,
                    experience_relevance=experience_relevance,
                    behavioral_fit=behavioral_fit,
                    platform_signal=platform_signal,
                    embedding_similarity=round(embedding_score, 2),
                    overall_score=overall,
                    rationale=llm_scores.get("rationale", "No rationale generated."),
                    skill_evidence=llm_scores.get("skill_evidence", []),
                    experience_evidence=llm_scores.get("experience_evidence", []),
                )
            )

        # Sort descending by overall_score, assign rank
        scored.sort(key=lambda x: x.overall_score, reverse=True)
        for i, s in enumerate(scored, start=1):
            s.rank = i

        return scored

    def _judge_candidate(
        self, role: RoleProfile, candidate: CandidateProfile
    ) -> dict:
        """Call LLM to score a single candidate; returns fallback scores on error."""
        if self._llm is None:
            return {
                "skill_match": 50,
                "experience_relevance": 50,
                "behavioral_fit": 50,
                "platform_signal": 50,
                "rationale": "LLM not available.",
                "skill_evidence": [],
                "experience_evidence": [],
            }
        prompt = _build_judge_prompt(role, candidate)
        try:
            return self._llm.call(prompt)
        except Exception as exc:
            logger.warning(
                "LLM judging failed for %r: %s — using neutral scores",
                candidate.name,
                exc,
            )
            return {
                "skill_match": 50,
                "experience_relevance": 50,
                "behavioral_fit": 50,
                "platform_signal": 50,
                "rationale": f"Automated scoring unavailable. Embedding similarity: {50:.1f}.",
                "skill_evidence": [],
                "experience_evidence": [],
            }
