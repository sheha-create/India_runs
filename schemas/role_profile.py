"""
schemas/role_profile.py — Pydantic model for the structured role profile
extracted from a raw job description.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RoleProfile(BaseModel):
    """Structured representation of a Job Description."""

    role_title: str = Field(default="Unknown Role", description="Job title / role name")
    seniority_level: str = Field(
        default="Unknown",
        description="e.g. Junior, Mid-level, Senior, Staff, Principal, Director"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Must-have technical and domain skills",
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Nice-to-have skills that differentiate candidates",
    )
    domain: Optional[str] = Field(
        default="",
        description="Industry or domain context (e.g. FinTech, Healthcare, E-commerce)",
    )
    years_experience_min: Optional[int] = Field(
        default=None,
        description="Minimum years of relevant experience required",
    )
    behavioral_expectations: List[str] = Field(
        default_factory=list,
        description="Soft-skill and cultural signals inferred from JD language "
                    "(e.g. 'fast-paced', 'cross-functional', 'ownership mentality')",
    )
    implicit_signals: List[str] = Field(
        default_factory=list,
        description="Implicit expectations not explicitly stated but inferred "
                    "(e.g. startup experience, open-source contributions, leadership under ambiguity)",
    )
    responsibilities_summary: Optional[str] = Field(
        default="",
        description="Short prose summary of main responsibilities",
    )

    @field_validator("domain", "responsibilities_summary", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v):
        """Convert None from LLM output to empty string."""
        return v if v is not None else ""

    def to_embedding_text(self) -> str:
        """Flatten the role profile to a rich text string for embedding."""
        parts = [
            f"Role: {self.role_title}",
            f"Seniority: {self.seniority_level}",
            f"Domain: {self.domain}",
            f"Required skills: {', '.join(self.required_skills)}",
            f"Preferred skills: {', '.join(self.preferred_skills)}",
            f"Experience: {self.years_experience_min}+ years" if self.years_experience_min else "",
            f"Behavioral expectations: {', '.join(self.behavioral_expectations)}",
            f"Implicit signals: {', '.join(self.implicit_signals)}",
            f"Responsibilities: {self.responsibilities_summary}",
        ]
        return "\n".join(p for p in parts if p)
