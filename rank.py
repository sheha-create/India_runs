"""
rank.py — Offline Candidate Ranking Engine (Hackathon Submission Script)

Produces the top-100 submission CSV from candidates.jsonl within the
5-minute CPU-only constraint (no external API calls).

Architecture:
  1. Load precomputed candidate embeddings (from precompute_embeddings.py)
  2. Embed the JD text locally (sentence-transformers, CPU)
  3. Compute cosine similarity (fast, vectorized)
  4. For every candidate, compute a deterministic feature score from their
     structured fields (skills, experience, signals, location, honeypot flags)
  5. Combine: final_score = α * embedding_sim + β * feature_score
  6. Apply honeypot penalty to push impossible profiles out of top 100
  7. Sort descending, take top 100, write submission CSV

Usage:
    # Requires pre-computed embeddings (run once):
    #   python precompute_embeddings.py
    #
    # Then produce submission:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

    # Or use the default paths:
    python rank.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
CANDIDATES_JSONL = os.path.join(
    "data",
    "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "candidates.jsonl",
)
EMBEDDINGS_NPY = os.path.join("data", "candidate_embeddings.npy")
IDS_JSON       = os.path.join("data", "candidate_ids.json")
JD_PATH        = os.path.join("data", "job_description.txt")
DEFAULT_OUT    = "submission.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Scoring weights ─────────────────────────────────────────────────────────────
# Must produce a non-increasing score column — weights control blend
ALPHA = 0.40   # embedding cosine similarity weight
BETA  = 0.60   # deterministic feature score weight

# ── Constants ───────────────────────────────────────────────────────────────────
COMPETITION_DATE = datetime(2026, 6, 17)
RECENT_ACTIVE_THRESHOLD_DAYS = 90   # last_active within 90 days = active

# ── JD Signals parsed from job_description.txt ─────────────────────────────────
# These are used for the deterministic feature scorer.
REQUIRED_SKILLS = {
    "python", "embeddings", "sentence-transformers", "sentence transformers",
    "vector database", "vector db", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch",
    "retrieval", "information retrieval", "ranking", "search", "ndcg", "mrr", "map",
    "rag", "retrieval-augmented generation",
    "nlp", "natural language processing",
    "llm", "large language model", "transformers", "bert", "gpt",
    "hybrid search", "dense retrieval", "semantic search",
    "learning to rank", "reranking", "re-ranking",
}

PREFERRED_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "lightgbm",
    "pytorch", "tensorflow", "jax",
    "bm25", "elasticsearch", "solr",
    "a/b testing", "ab testing", "evaluation framework",
    "kubernetes", "docker", "spark", "kafka",
    "distributed systems", "ml ops", "mlops",
    "hr tech", "hr-tech", "recruiting tech", "talent intelligence",
    "open source", "github",
}

# Titles that score highly for this JD
HIGH_FIT_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "applied ml", "applied ai",
    "search engineer", "search relevance engineer",
    "nlp engineer", "nlp scientist", "nlp researcher",
    "data scientist", "senior data scientist", "staff data scientist",
    "research engineer", "ml researcher",
    "backend engineer", "software engineer",  # ok but not great
    "founding engineer", "staff engineer", "principal engineer",
    "ranking engineer", "recommendation engineer", "recommendation systems",
    "retrieval engineer", "information retrieval",
    "mlops", "ml platform", "ai platform",
}

MID_FIT_TITLES = {
    "data engineer", "platform engineer", "infrastructure engineer",
    "full stack engineer", "full-stack engineer",
    "tech lead", "engineering lead", "engineering manager",
    "solutions architect",
}

ANTI_FIT_TITLES = {
    "marketing manager", "operations manager", "business analyst",
    "product manager", "customer support", "accountant", "hr manager",
    "content writer", "seo", "project manager", "scrum master",
    "sales", "finance", "legal", "graphic designer", "designer",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "hardware engineer", "embedded engineer",
    "hr", "recruiter", "operations", "supply chain", "logistics",
    "quality assurance manager", "compliance", "auditor",
}

# Consulting-only career companies per JD (disqualifier if ENTIRE career is only here)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "mphasis", "mindtree",
    "tech mahindra", "hexaware", "l&t infotech", "ltimindtree",
}

# Indian locations that get a boost (per JD)
PREFERRED_LOCATIONS = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurgaon", "gurugram", "ncr",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Honeypot Detection
# ══════════════════════════════════════════════════════════════════════════════

def is_honeypot(cand: dict) -> bool:
    """Return True if candidate has an impossible / fabricated profile."""
    profile = cand.get("profile", {})
    career  = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})
    edu     = cand.get("education", [])
    skills  = cand.get("skills", [])

    # Rule 1: Salary inversion (min > max)
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal and sal.get("min", 0) > sal.get("max", float("inf")):
        return True

    # Rule 2: Full-time career started ≥5 years before any college start
    if edu:
        min_edu_start = min((e.get("start_year", 9999) for e in edu), default=9999)
        for job in career:
            sd    = job.get("start_date")
            title = job.get("title", "").lower()
            # Skip internships, training programmes
            if sd and not any(x in title for x in ("intern", "trainee", "student", "apprentice")):
                try:
                    sy = datetime.strptime(sd, "%Y-%m-%d").year
                    if sy < min_edu_start - 5:
                        return True
                except Exception:
                    pass

    # Rule 3: Job duration_months massively inconsistent with actual dates (>24 mo off)
    for job in career:
        sd  = job.get("start_date")
        ed  = job.get("end_date")
        dur = job.get("duration_months", 0)
        if sd:
            try:
                s_dt = datetime.strptime(sd, "%Y-%m-%d")
                e_dt = datetime.strptime(ed, "%Y-%m-%d") if ed else COMPETITION_DATE
                diff = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                if abs(diff - dur) > 24:
                    return True
            except Exception:
                pass

    # Rule 4: Stated YOE extremely low vs career date-span OR career much exceeds YOE
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(j.get("duration_months", 0) for j in career)
    total_yrs    = total_months / 12.0
    if yoe > 5 and total_yrs < 1.0:
        return True
    if total_yrs > yoe + 8.0:   # career claims >8 extra years
        return True

    # Rule 5: Invalid platform rates
    for field in ("recruiter_response_rate", "interview_completion_rate"):
        val = signals.get(field)
        if val is not None and (val < 0.0 or val > 1.0):
            return True
    oar = signals.get("offer_acceptance_rate")
    if oar is not None and (oar < -1.0 or oar > 1.0):
        return True
    gas = signals.get("github_activity_score")
    if gas is not None and (gas < -1.0 or gas > 100.0):
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Deterministic Feature Scorer
# ══════════════════════════════════════════════════════════════════════════════

def _skill_overlap(cand_skills: List[dict], target_set: set) -> float:
    """Fraction of target skills matched by the candidate (capped at 1.0)."""
    if not cand_skills or not target_set:
        return 0.0
    matched = sum(
        1 for s in cand_skills
        if any(t in s.get("name", "").lower() for t in target_set)
    )
    return min(matched / max(len(target_set) * 0.15, 1), 1.0)  # normalize loosely


def _title_score(title: str) -> float:
    t = title.lower()
    if any(h in t for h in HIGH_FIT_TITLES):
        return 1.0
    if any(m in t for m in MID_FIT_TITLES):
        return 0.5
    if any(a in t for a in ANTI_FIT_TITLES):
        return 0.0
    # For truly unknown titles, default to a low score
    # rather than neutral — the JD explicitly describes a narrow profile
    return 0.15


def _experience_score(yoe: float) -> float:
    """Score YOE — 5-9 years is the sweet spot for this JD."""
    if yoe < 1:
        return 0.05
    if yoe < 3:
        return 0.3
    if yoe < 5:
        return 0.6
    if yoe <= 9:
        return 1.0
    if yoe <= 12:
        return 0.85
    return 0.65  # Very senior — still valuable but not ideal fit


def _consulting_only_penalty(career: List[dict]) -> float:
    """Return a penalty multiplier if career is entirely at consulting firms."""
    if not career:
        return 1.0
    non_consulting = sum(
        1 for j in career
        if not any(c in j.get("company", "").lower() for c in CONSULTING_FIRMS)
    )
    if non_consulting == 0:
        return 0.25  # strong penalty for consulting-only career
    return 1.0


def _platform_signal_score(signals: dict) -> float:
    """Score the Redrob platform engagement signals."""
    score = 0.0

    # Active in last 90 days
    last_active = signals.get("last_active_date", "")
    try:
        lad = datetime.strptime(last_active, "%Y-%m-%d")
        days_since = (COMPETITION_DATE - lad).days
        if days_since < 30:
            score += 0.30
        elif days_since < 90:
            score += 0.20
        elif days_since < 180:
            score += 0.10
    except Exception:
        pass

    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 0.20

    # Recruiter response rate
    rrr = signals.get("recruiter_response_rate", 0) or 0
    score += rrr * 0.15

    # Notice period
    np_days = signals.get("notice_period_days", 90)
    if np_days <= 30:
        score += 0.15
    elif np_days <= 60:
        score += 0.08
    elif np_days <= 90:
        score += 0.03

    # GitHub activity
    gas = signals.get("github_activity_score", -1) or -1
    if gas > 0:
        score += min(gas / 100.0, 1.0) * 0.10

    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0) or 0
    score += icr * 0.05

    # Profile completeness
    pcs = signals.get("profile_completeness_score", 0) or 0
    score += (pcs / 100.0) * 0.05

    return min(score, 1.0)


def _location_score(location: str, country: str) -> float:
    loc = (location + " " + country).lower()
    if any(p in loc for p in PREFERRED_LOCATIONS):
        return 1.0
    if "india" in loc:
        return 0.7
    return 0.3  # outside India — still possible per JD but lower pref


def feature_score(cand: dict) -> float:
    """
    Compute a deterministic 0-1 feature score for a candidate.
    Weights are calibrated to the JD requirements.
    """
    profile = cand.get("profile", {})
    skills  = cand.get("skills", [])
    career  = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})

    yoe           = float(profile.get("years_of_experience", 0) or 0)
    current_title = profile.get("current_title", "")
    location      = profile.get("location", "")
    country       = profile.get("country", "")

    # Component scores
    s_title    = _title_score(current_title)
    s_exp      = _experience_score(yoe)
    s_req_skill= _skill_overlap(skills, REQUIRED_SKILLS)
    s_pref_skill= _skill_overlap(skills, PREFERRED_SKILLS)
    s_skill    = 0.7 * s_req_skill + 0.3 * s_pref_skill
    s_platform = _platform_signal_score(signals)
    s_location = _location_score(location, country)
    c_penalty   = _consulting_only_penalty(career)

    # Weighted combination
    raw = (
        0.30 * s_title
        + 0.20 * s_exp
        + 0.25 * s_skill
        + 0.15 * s_platform
        + 0.10 * s_location
    )
    score = raw * c_penalty

    # Relevance gate: if title AND skills are both zero, cap at 0.10
    # to prevent irrelevant profiles from floating up on platform signals alone
    if s_title == 0.0 and s_req_skill == 0.0:
        score = min(score, 0.10)

    return score


def build_reasoning(cand: dict, rank: int, emb_score: float, feat_score: float) -> str:
    """Build a specific, honest 1-2 sentence reasoning for this candidate."""
    profile = cand.get("profile", {})
    skills  = cand.get("skills", [])
    career  = cand.get("career_history", [])
    signals = cand.get("redrob_signals", {})

    yoe           = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "Unknown Title")
    location      = profile.get("location", "")
    country       = profile.get("country", "")

    # Collect matched relevant skills
    top_skills = [
        s["name"] for s in skills
        if any(t in s["name"].lower() for t in REQUIRED_SKILLS | PREFERRED_SKILLS)
    ][:5]

    # Notice period
    np_days = signals.get("notice_period_days", None)
    np_note = ""
    if np_days is not None:
        if np_days <= 30:
            np_note = f"notice period of {np_days}d"
        elif np_days > 90:
            np_note = f"notice period concern ({np_days}d)"

    # Active status
    last_active = signals.get("last_active_date", "")
    active_note = ""
    try:
        lad = datetime.strptime(last_active, "%Y-%m-%d")
        days_since = (COMPETITION_DATE - lad).days
        if days_since > 180:
            active_note = f"last active {days_since}d ago — availability uncertain"
        elif days_since < 30:
            active_note = "recently active"
    except Exception:
        pass

    # Companies
    companies = [j.get("company", "") for j in career[:2] if j.get("company")]
    company_str = " → ".join(companies) if companies else ""

    # Location fit
    loc_str = f"{location}, {country}" if location else country

    # Build sentence 1: what makes them fit
    skill_str = ", ".join(top_skills) if top_skills else "adjacent skills"
    sentence1 = (
        f"{yoe:.0f} yrs exp as {current_title}"
        f"{(' at ' + company_str) if company_str else ''}"
        f"; relevant skills: {skill_str}."
    )

    # Build sentence 2: availability/location and any concerns
    parts = [x for x in [loc_str, np_note, active_note] if x]
    if rank <= 20:
        sentence2 = "Strong alignment with JD's production AI/retrieval requirements. "
    elif rank <= 50:
        sentence2 = "Moderate fit; profile suggests relevant but not directly matching experience. "
    else:
        sentence2 = "Adjacent profile; included based on partial skill overlap and availability. "

    if parts:
        sentence2 += " | ".join(parts) + "."

    return (sentence1 + " " + sentence2).strip()[:400]


# ══════════════════════════════════════════════════════════════════════════════
#  Main Ranking Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_embeddings(emb_path: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    print(f"Loading precomputed embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    with open(ids_path, "r") as f:
        cids = json.load(f)
    print(f"  Loaded {len(cids)} candidate embeddings, shape={embeddings.shape}")
    return embeddings, cids


def embed_jd(jd_text: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore
    print(f"Embedding JD text locally with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode([jd_text], convert_to_numpy=True, show_progress_bar=False)
    print("  JD embedded.")
    return emb  # shape (1, dim)


def load_candidates_metadata(jsonl_path: str) -> Dict[str, dict]:
    """Stream candidates.jsonl and load raw dicts keyed by candidate_id."""
    print(f"Loading candidate metadata from {jsonl_path}...")
    cands: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            cands[obj["candidate_id"]] = obj
            if (i + 1) % 20000 == 0:
                print(f"  ... loaded {i+1} candidates")
    print(f"  Total loaded: {len(cands)}")
    return cands


def rank_candidates(
    candidates_jsonl: str,
    embeddings_npy: str,
    ids_json: str,
    jd_path: str,
    output_path: str,
    top_n: int = 100,
) -> None:
    t0 = time.time()

    # 1. Load precomputed candidate embeddings
    cand_embeddings, cand_ids = load_embeddings(embeddings_npy, ids_json)

    # 2. Embed the JD
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
    jd_embedding = embed_jd(jd_text)  # shape (1, dim)

    # 3. Cosine similarity (vectorized)
    print("Computing cosine similarities...")
    # Normalize
    cand_norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1e-9
    cand_normed = cand_embeddings / cand_norms

    jd_norm = np.linalg.norm(jd_embedding)
    jd_normed = jd_embedding / (jd_norm if jd_norm > 0 else 1e-9)

    cos_sims = (cand_normed @ jd_normed.T).flatten()  # shape (N,)
    cos_sims_scaled = (cos_sims.clip(0, 1) * 100)  # 0-100

    # 4. Load candidate metadata (for feature scoring + honeypot detection)
    cand_data = load_candidates_metadata(candidates_jsonl)

    # 5. Compute feature scores and honeypot flags
    print("Computing feature scores and detecting honeypots...")
    results = []
    for idx, cid in enumerate(cand_ids):
        cand = cand_data.get(cid)
        if cand is None:
            continue

        emb_score = float(cos_sims_scaled[idx])

        # Hard filter: honeypots get a massive score penalty (effectively removed from top 100)
        if is_honeypot(cand):
            results.append({
                "candidate_id": cid,
                "emb_score": emb_score,
                "feat_score": 0.0,
                "final_score": 0.0,  # pushed to bottom
                "is_honeypot": True,
            })
            continue

        fscore = feature_score(cand)
        final  = ALPHA * (emb_score / 100.0) + BETA * fscore  # in [0,1]
        final_scaled = round(final * 100, 4)

        results.append({
            "candidate_id": cid,
            "emb_score": emb_score,
            "feat_score": round(fscore * 100, 4),
            "final_score": final_scaled,
            "is_honeypot": False,
        })

    # 6. Sort descending by final_score, break ties by candidate_id ascending
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # 7. Ensure scores are non-increasing (submission constraint)
    prev_score = results[0]["final_score"] if results else 0
    for r in results:
        if r["final_score"] > prev_score:
            r["final_score"] = prev_score
        prev_score = r["final_score"]

    # 8. Take top 100 and assign ranks 1-100
    top100 = results[:top_n]

    # 9. Write submission CSV
    print(f"Writing submission CSV to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank_i, row in enumerate(top100, start=1):
            cid    = row["candidate_id"]
            score  = row["final_score"]
            cand   = cand_data.get(cid, {})
            reason = build_reasoning(cand, rank_i, row["emb_score"], row["feat_score"])
            # Escape commas in reasoning
            reason_escaped = '"' + reason.replace('"', "'") + '"'
            f.write(f"{cid},{rank_i},{score:.6f},{reason_escaped}\n")

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"   Submission saved → {output_path}")
    print(f"   Honeypots in top-100: {sum(1 for r in top100 if r['is_honeypot'])}")
    print(f"   Score range: {top100[0]['final_score']:.4f} → {top100[-1]['final_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline candidate ranker — produces hackathon submission CSV."
    )
    parser.add_argument(
        "--candidates",
        default=CANDIDATES_JSONL,
        help=f"Path to candidates.jsonl (default: {CANDIDATES_JSONL})",
    )
    parser.add_argument(
        "--embeddings",
        default=EMBEDDINGS_NPY,
        help=f"Path to precomputed embeddings .npy (default: {EMBEDDINGS_NPY})",
    )
    parser.add_argument(
        "--ids",
        default=IDS_JSON,
        help=f"Path to candidate IDs JSON (default: {IDS_JSON})",
    )
    parser.add_argument(
        "--jd",
        default=JD_PATH,
        help=f"Path to job description text file (default: {JD_PATH})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output submission CSV path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    # Validate required files exist
    missing = []
    for label, path in [
        ("candidates jsonl", args.candidates),
        ("embeddings npy",   args.embeddings),
        ("candidate ids",    args.ids),
        ("job description",  args.jd),
    ]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print(m)
        if any("embeddings" in m or "ids" in m for m in missing):
            print("\nHave you run precompute_embeddings.py first?")
            print("  python precompute_embeddings.py")
        sys.exit(1)

    rank_candidates(
        candidates_jsonl=args.candidates,
        embeddings_npy=args.embeddings,
        ids_json=args.ids,
        jd_path=args.jd,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
