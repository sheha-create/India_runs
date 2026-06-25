"""
config.py — Centralized configuration for AI Candidate Ranking System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider (Groq) ────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Groq model — llama-3.3-70b-versatile is fast, supports JSON mode
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# LLM temperature — 0 for reproducibility
LLM_TEMPERATURE: float = 0.0

# ── Embedding Model ────────────────────────────────────────────────────────────
# Downloaded once, cached locally under ~/.cache/huggingface
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Pipeline Settings ─────────────────────────────────────────────────────────
# Number of top candidates (by embedding score) passed to the LLM reasoning pass
TOP_N_FOR_LLM: int = 20

# ── Scoring Weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHTS = {
    "skill_match":            0.35,
    "experience_relevance":   0.25,
    "behavioral_fit":         0.15,
    "platform_signal":        0.10,
    "embedding_similarity":   0.15,
}

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "data/output"
OUTPUT_CSV: str = f"{OUTPUT_DIR}/ranked_candidates.csv"
OUTPUT_JSON: str = f"{OUTPUT_DIR}/ranked_candidates.json"
