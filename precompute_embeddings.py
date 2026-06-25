"""
precompute_embeddings.py — Pre-computation step (run once before rank.py).

Reads the 100K candidate pool, builds a rich text representation of each
candidate, and encodes it with sentence-transformers/all-MiniLM-L6-v2 on CPU.

Outputs:
  data/candidate_embeddings.npy  — float32 matrix (100000, 384)
  data/candidate_ids.json        — ordered list of candidate_ids

Estimated runtime on CPU: 8-15 minutes.
The ranking step (rank.py) then runs in under 2 minutes.

Usage:
    python precompute_embeddings.py
"""
import json
import os
import time

import numpy as np
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
LARGE_FILE = os.path.join(
    "data",
    "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge",
    "candidates.jsonl",
)
EMBEDDINGS_OUT = os.path.join("data", "candidate_embeddings.npy")
IDS_OUT        = os.path.join("data", "candidate_ids.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_candidate_text(cand: dict) -> str:
    """
    Build a rich, JD-aligned text representation of a candidate.
    Weights content toward what matters for the 'Senior AI Engineer' JD.
    """
    prof   = cand.get("profile", {})
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    edu    = cand.get("education", [])
    certs  = cand.get("certifications", [])
    signals= cand.get("redrob_signals", {})

    # Profile basics
    title   = prof.get("current_title", "")
    headline= prof.get("headline", "")
    summary = prof.get("summary", "")[:400]  # truncate
    yoe     = prof.get("years_of_experience", 0)
    location= prof.get("location", "")
    country = prof.get("country", "")

    # Skills — include name, proficiency, and duration for richer signal
    skill_parts = []
    for s in skills:
        name = s.get("name", "")
        prof_level = s.get("proficiency", "")
        dur  = s.get("duration_months", 0)
        if name:
            skill_parts.append(f"{name} ({prof_level}, {dur}mo)")
    skills_str = ", ".join(skill_parts)

    # Career history — recent 3 roles
    career_parts = []
    for job in career[:3]:
        company  = job.get("company", "")
        job_title= job.get("title", "")
        industry = job.get("industry", "")
        desc     = job.get("description", "")[:200]
        dur_m    = job.get("duration_months", 0)
        career_parts.append(
            f"{job_title} at {company} ({industry}, {dur_m}mo): {desc}"
        )
    career_str = " | ".join(career_parts)

    # Education
    edu_parts = []
    for e in edu:
        inst  = e.get("institution", "")
        deg   = e.get("degree", "")
        field = e.get("field_of_study", "")
        tier  = e.get("tier", "")
        edu_parts.append(f"{deg} in {field} from {inst} ({tier})")
    edu_str = " | ".join(edu_parts)

    # Certifications
    cert_str = ", ".join(
        c.get("name", "") for c in certs if c.get("name")
    )

    # Platform signals (availability proxy)
    open_to_work = "open to work" if signals.get("open_to_work_flag") else ""
    github_score = signals.get("github_activity_score", -1) or -1
    github_str   = f"github_activity={github_score:.0f}" if github_score >= 0 else ""
    notice       = signals.get("notice_period_days", None)
    notice_str   = f"notice={notice}days" if notice is not None else ""

    parts = [
        f"Title: {title}",
        f"Headline: {headline}",
        f"Experience: {yoe} years",
        f"Location: {location}, {country}",
        f"Summary: {summary}",
        f"Skills: {skills_str}",
        f"Career: {career_str}",
        f"Education: {edu_str}",
    ]
    if cert_str:
        parts.append(f"Certifications: {cert_str}")
    for sig in [open_to_work, github_str, notice_str]:
        if sig:
            parts.append(sig)

    return "\n".join(p for p in parts if p)


def main():
    if not os.path.exists(LARGE_FILE):
        print(f"ERROR: candidates.jsonl not found at:\n  {LARGE_FILE}")
        return

    print(f"Loading sentence-transformers model ({EMBEDDING_MODEL})...")
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Reading and building candidate texts from candidates.jsonl...")
    t0 = time.time()
    texts = []
    cids  = []

    with open(LARGE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=100000, desc="Parsing JSONL", unit="cand"):
            if not line.strip():
                continue
            cand = json.loads(line)
            cids.append(cand["candidate_id"])
            texts.append(build_candidate_text(cand))

    parse_time = time.time() - t0
    print(f"Parsed {len(texts)} candidates in {parse_time:.1f}s")

    print(f"Generating embeddings (batch_size=256, CPU)...")
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embed_time = time.time() - t1
    print(f"Embedding took {embed_time:.1f}s — shape: {embeddings.shape}")

    print(f"Saving embeddings to {EMBEDDINGS_OUT} ...")
    np.save(EMBEDDINGS_OUT, embeddings.astype(np.float32))

    print(f"Saving candidate IDs to {IDS_OUT} ...")
    with open(IDS_OUT, "w", encoding="utf-8") as f:
        json.dump(cids, f)

    total = time.time() - t0
    print(f"\n✅ Pre-computation complete in {total:.1f}s")
    print(f"   Embeddings: {EMBEDDINGS_OUT}  ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"   IDs:        {IDS_OUT}")
    print(f"\nNext step — produce submission:")
    print(f"  python rank.py --out submission.csv")


if __name__ == "__main__":
    main()
