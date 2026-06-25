"""
schemas/candidate_profile.py — Pydantic model for a normalized candidate profile.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str = ""
    title: str = ""
    duration_years: Optional[float] = None
    description: str = ""
    domain: str = ""


class CandidateProfile(BaseModel):
    """Normalized representation of a single candidate."""

    # ── Identity ───────────────────────────────────────────────
    candidate_id: str
    name: str
    email: str = ""
    current_title: str = ""

    # ── Skills ─────────────────────────────────────────────────
    explicit_skills: List[str] = Field(
        default_factory=list,
        description="Skills explicitly listed in profile/resume",
    )
    inferred_skills: List[str] = Field(
        default_factory=list,
        description="Skills inferred by LLM from work descriptions and project text",
    )

    # ── Experience ─────────────────────────────────────────────
    total_years_experience: float = 0.0
    work_history: List[WorkExperience] = Field(default_factory=list)
    career_trajectory: str = Field(
        default="",
        description="LLM-generated narrative of career progression pattern",
    )
    domain_exposure: List[str] = Field(
        default_factory=list,
        description="Industries / domains the candidate has worked in",
    )

    # ── Behavioral / Soft Signals ──────────────────────────────
    behavioral_signals: List[str] = Field(
        default_factory=list,
        description="Behavioral indicators extracted from profile text "
                    "(e.g. led cross-functional team, delivered under tight deadlines)",
    )

    # ── Platform Signals ───────────────────────────────────────
    platform_signals: List[str] = Field(
        default_factory=list,
        description="GitHub/LinkedIn/other platform indicators "
                    "(e.g. active open-source contributor, 500+ connections)",
    )

    # ── Raw data (for transparency) ────────────────────────────
    raw_summary: str = ""
    raw_projects: str = ""
    education: str = ""

    def to_embedding_text(self) -> str:
        """Flatten the candidate profile to a rich text string for embedding."""
        all_skills = list(set(self.explicit_skills + self.inferred_skills))
        parts = [
            f"Name: {self.name}",
            f"Current role: {self.current_title}",
            f"Total experience: {self.total_years_experience} years",
            f"Skills: {', '.join(all_skills)}",
            f"Domain exposure: {', '.join(self.domain_exposure)}",
            f"Career trajectory: {self.career_trajectory}",
            f"Behavioral signals: {', '.join(self.behavioral_signals)}",
            f"Platform signals: {', '.join(self.platform_signals)}",
            f"Education: {self.education}",
            f"Summary: {self.raw_summary}",
            f"Projects: {self.raw_projects}",
        ]
        return "\n".join(p for p in parts if p)


class ScoredCandidate(BaseModel):
    """A candidate profile with ranking scores and rationale."""

    profile: CandidateProfile

    # ── Sub-scores (0–100) ─────────────────────────────────────
    skill_match: float = 0.0
    experience_relevance: float = 0.0
    behavioral_fit: float = 0.0
    platform_signal: float = 0.0
    embedding_similarity: float = 0.0

    # ── Final composite score ──────────────────────────────────
    overall_score: float = 0.0

    # ── Explainability ─────────────────────────────────────────
    rationale: str = ""
    skill_evidence: List[str] = Field(default_factory=list)
    experience_evidence: List[str] = Field(default_factory=list)

    # ── Rank ───────────────────────────────────────────────────
    rank: int = 0
