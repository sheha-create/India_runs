"""
modules/output_writer.py — Shortlist Generation Module (PRD §7.4)

Serializes the ranked candidate list to CSV and JSON output files.
Both formats include all scoring dimensions and rationale for full auditability.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

import pandas as pd

import config
from schemas.candidate_profile import ScoredCandidate

logger = logging.getLogger(__name__)

# Column order for the output CSV (recruiter-friendly)
CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "name",
    "current_title",
    "total_years_experience",
    "overall_score",
    "skill_match",
    "experience_relevance",
    "behavioral_fit",
    "platform_signal",
    "embedding_similarity",
    "email",
    "skill_evidence",
    "experience_evidence",
    "rationale",
]


def _to_record(sc: ScoredCandidate) -> dict:
    p = sc.profile
    return {
        "rank": sc.rank,
        "candidate_id": p.candidate_id,
        "name": p.name,
        "email": p.email,
        "current_title": p.current_title,
        "total_years_experience": p.total_years_experience,
        "overall_score": sc.overall_score,
        "skill_match": sc.skill_match,
        "experience_relevance": sc.experience_relevance,
        "behavioral_fit": sc.behavioral_fit,
        "platform_signal": sc.platform_signal,
        "embedding_similarity": sc.embedding_similarity,
        "explicit_skills": "; ".join(p.explicit_skills),
        "inferred_skills": "; ".join(p.inferred_skills),
        "domain_exposure": "; ".join(p.domain_exposure),
        "behavioral_signals": "; ".join(p.behavioral_signals),
        "platform_signals": "; ".join(p.platform_signals),
        "career_trajectory": p.career_trajectory,
        "education": p.education,
        "skill_evidence": "; ".join(sc.skill_evidence),
        "experience_evidence": "; ".join(sc.experience_evidence),
        "rationale": sc.rationale,
    }


class OutputWriter:
    """Writes ranked candidates to CSV and JSON."""

    def __init__(
        self,
        output_dir: str = config.OUTPUT_DIR,
        csv_path: str = config.OUTPUT_CSV,
        json_path: str = config.OUTPUT_JSON,
    ) -> None:
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.json_path = json_path
        os.makedirs(output_dir, exist_ok=True)

    def write(self, scored_candidates: List[ScoredCandidate]) -> None:
        """Write both CSV and JSON output files."""
        records = [_to_record(sc) for sc in scored_candidates]
        self._write_csv(records)
        self._write_json(scored_candidates)
        logger.info(
            "Output written -> CSV: %s | JSON: %s",
            self.csv_path,
            self.json_path,
        )

    def _write_csv(self, records: list) -> None:
        df = pd.DataFrame(records)
        # Reorder columns (keep extras at end)
        ordered_cols = [c for c in CSV_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra_cols]
        df.to_csv(self.csv_path, index=False, encoding="utf-8")
        logger.info("CSV saved: %s (%d rows)", self.csv_path, len(df))

    def _write_json(self, scored_candidates: List[ScoredCandidate]) -> None:
        output = []
        for sc in scored_candidates:
            p = sc.profile
            entry = {
                "rank": sc.rank,
                "candidate_id": p.candidate_id,
                "name": p.name,
                "email": p.email,
                "current_title": p.current_title,
                "total_years_experience": p.total_years_experience,
                "scores": {
                    "overall": sc.overall_score,
                    "skill_match": sc.skill_match,
                    "experience_relevance": sc.experience_relevance,
                    "behavioral_fit": sc.behavioral_fit,
                    "platform_signal": sc.platform_signal,
                    "embedding_similarity": sc.embedding_similarity,
                },
                "skills": {
                    "explicit": p.explicit_skills,
                    "inferred": p.inferred_skills,
                },
                "career_trajectory": p.career_trajectory,
                "domain_exposure": p.domain_exposure,
                "behavioral_signals": p.behavioral_signals,
                "platform_signals": p.platform_signals,
                "education": p.education,
                "skill_evidence": sc.skill_evidence,
                "experience_evidence": sc.experience_evidence,
                "rationale": sc.rationale,
            }
            output.append(entry)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("JSON saved: %s (%d entries)", self.json_path, len(output))

    def print_summary(self, scored_candidates: List[ScoredCandidate], top_n: int = 10) -> None:
        """Print a pretty console summary of top-N ranked candidates."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(
                title=f"🏆 Top {min(top_n, len(scored_candidates))} Ranked Candidates",
                box=box.ROUNDED,
                show_lines=True,
                header_style="bold cyan",
            )
            table.add_column("Rank", style="bold yellow", justify="center", width=6)
            table.add_column("Name", style="bold white", width=22)
            table.add_column("Title", style="dim", width=26)
            table.add_column("Score", justify="center", style="bold green", width=8)
            table.add_column("Skill", justify="center", width=7)
            table.add_column("Exp", justify="center", width=7)
            table.add_column("Behav", justify="center", width=7)
            table.add_column("Rationale (excerpt)", width=45)

            for sc in scored_candidates[:top_n]:
                rationale_short = sc.rationale[:120] + "…" if len(sc.rationale) > 120 else sc.rationale
                table.add_row(
                    f"#{sc.rank}",
                    sc.profile.name,
                    sc.profile.current_title[:24],
                    f"{sc.overall_score:.1f}",
                    f"{sc.skill_match:.0f}",
                    f"{sc.experience_relevance:.0f}",
                    f"{sc.behavioral_fit:.0f}",
                    rationale_short,
                )
            console.print(table)
        except Exception:
            # Fallback plain text
            print(f"\n{'='*60}")
            print(f"TOP {min(top_n, len(scored_candidates))} RANKED CANDIDATES")
            print(f"{'='*60}")
            for sc in scored_candidates[:top_n]:
                try:
                    name_clean = sc.profile.name.encode('ascii', 'ignore').decode('ascii')
                except Exception:
                    name_clean = "Candidate"
                print(
                    f"#{sc.rank:2d} | {name_clean:<25} | "
                    f"Score: {sc.overall_score:5.1f} | "
                    f"Skill: {sc.skill_match:4.0f} | "
                    f"Exp: {sc.experience_relevance:4.0f}"
                )
            print(f"{'='*60}\n")
