"""
main.py — CLI Entry Point for AI Candidate Ranking System.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import typer
from dotenv import load_dotenv

# Load env variables before other imports to ensure key configuration is loaded
load_dotenv()


import config
from modules.jd_parser import JDParser
from modules.candidate_profiler import CandidateProfiler
from modules.embedder import Embedder
from modules.ranker import Ranker
from modules.output_writer import OutputWriter

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ranking_pipeline")

app = typer.Typer(help="AI Candidate Ranking System CLI")

@app.command()
def run(
    jd_path: str = typer.Option(
        ..., "--jd", "-j", help="Path to the Job Description text file."
    ),
    candidates_path: Optional[str] = typer.Option(
        None, "--candidates", "-c", help="Path to the Candidate CSV or JSONL dataset file. Defaults to public candidates.jsonl."
    ),
    top_n: int = typer.Option(
        config.TOP_N_FOR_LLM, "--top-n", "-n", help="Number of candidates to pass to LLM reasoning."
    ),
    use_llm_profiling: bool = typer.Option(
        True, "--llm-profile/--no-llm-profile", help="Enable or disable LLM-based parsing of individual candidates."
    ),
):
    """
    Run the AI candidate ranking pipeline:
    1. Parse job description to structured JSON.
    2. Load and profile candidates.
    3. Run first-pass fast embedding ranking.
    4. Run LLM reasoning pass on top-N candidates.
    5. Aggregate scores and write structured outputs (CSV & JSON).
    """
    # Resolve candidates_path if not provided
    if candidates_path is None:
        candidates_path = os.path.join(
            "data",
            "[PUB] India_runs_data_and_ai_challenge",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "candidates.jsonl",
        )

    # 1. Verification of inputs
    if not os.path.exists(jd_path):
        logger.error("Job description file not found: %s", jd_path)
        raise typer.Exit(code=1)
    if not os.path.exists(candidates_path):
        logger.error("Candidates file not found: %s", candidates_path)
        raise typer.Exit(code=1)

    # API Key check
    if not config.GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set. Please create a .env file from .env.example and populate your key.")
        raise typer.Exit(code=1)

    logger.info("Starting candidate ranking pipeline using Groq model: %s", config.GROQ_MODEL)

    # Load input data
    import pandas as pd
    import json
    from modules.candidate_profiler import map_jsonl_to_flat_dict
    try:
        if candidates_path.lower().endswith(".jsonl"):
            logger.info("Loading candidates from JSONL file: %s", candidates_path)
            raw_cands = []
            with open(candidates_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    raw_cands.append(json.loads(line))
            flat_rows = [map_jsonl_to_flat_dict(c) for c in raw_cands]
            candidates_df = pd.DataFrame(flat_rows)
            logger.info("Successfully loaded and flattened %d candidate rows from JSONL", len(candidates_df))
        else:
            candidates_df = pd.read_csv(candidates_path)
            logger.info("Successfully loaded %d raw candidate rows from CSV: %s", len(candidates_df), candidates_path)
    except Exception as e:
        logger.error("Failed to read candidates dataset file: %s", e)
        raise typer.Exit(code=1)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # 2. Pipeline Steps
    # Step 2.1: Parse JD
    jd_parser = JDParser()
    role_profile = jd_parser.parse(jd_text)

    # Step 2.2: Profile candidates
    profiler = CandidateProfiler(use_llm=use_llm_profiling)
    candidate_profiles = profiler.profile_dataframe(candidates_df)

    # Step 2.3: First-pass embedding ranking
    embedder = Embedder()
    embedding_ranked = embedder.rank_candidates(role_profile, candidate_profiles)

    # Filter to Top-N for the detailed LLM pass
    logger.info("Filtering to top %d candidates for LLM reasoning pass", top_n)
    top_candidates = embedding_ranked[:top_n]
    remaining_candidates = embedding_ranked[top_n:]

    # Step 2.4: Second-pass LLM judging and score aggregation
    ranker = Ranker()
    scored_top_candidates = ranker.rank(role_profile, top_candidates)

    # For candidates who did not make the top-N cutoff, provide baseline scores
    # based entirely on their embedding similarity.
    all_scored_candidates = list(scored_top_candidates)
    
    # Starting rank for the remaining candidates
    current_rank = len(all_scored_candidates) + 1
    for cand, emb_score in remaining_candidates:
        # Build a placeholder scored entry
        from schemas.candidate_profile import ScoredCandidate
        scored_cand = ScoredCandidate(
            profile=cand,
            skill_match=emb_score, # Use embedding similarity as baseline
            experience_relevance=emb_score,
            behavioral_fit=50.0, # Neutral
            platform_signal=50.0, # Neutral
            embedding_similarity=round(emb_score, 2),
            overall_score=round(emb_score, 2),
            rationale=f"Ranked outside the top {top_n} pre-filter. Embedding similarity score: {emb_score:.1f}.",
            rank=current_rank
        )
        all_scored_candidates.append(scored_cand)
        current_rank += 1

    # Step 2.5: Write Outputs
    writer = OutputWriter()
    writer.write(all_scored_candidates)
    writer.print_summary(all_scored_candidates, top_n=min(10, len(all_scored_candidates)))

    logger.info("Pipeline complete! Output files generated in %s/", config.OUTPUT_DIR)

if __name__ == "__main__":
    app()
