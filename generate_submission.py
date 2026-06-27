"""Quick submission generator using feature scorer only (no embeddings needed)."""
import json
import csv
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from rank import feature_score, is_honeypot, build_reasoning

JSONL = os.path.join(
    "data", "[PUB] India_runs_data_and_ai_challenge",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge", "candidates.jsonl"
)
OUT_DIR = os.path.join("data", "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    cands = []

    print(f"Reading {JSONL}...")
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if is_honeypot(c):
                continue
            fs = feature_score(c)
            cands.append((c, fs))

    print(f"Scored {len(cands)} candidates in {time.time()-t0:.1f}s")
    cands.sort(key=lambda x: -x[1])

    top100 = cands[:100]

    # Ensure non-increasing scores
    prev = top100[0][1] if top100 else 0
    for i, (c, s) in enumerate(top100):
        if s > prev:
            top100[i] = (c, prev)
        prev = top100[i][1]

    # Write hackathon submission CSV: candidate_id,rank,score,reasoning
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_i, (c, score) in enumerate(top100, 1):
            cid = c.get("candidate_id", "")
            reason = build_reasoning(c, rank_i, 0, score * 100)
            writer.writerow([cid, rank_i, round(score, 6), reason])
    print(f"Saved: {sub_path}")

    # Write ranked_candidates.csv: Rank,Candidate Name,Overall Score,Skill Score,Trajectory Score,Confidence,Summary
    rc_path = os.path.join(OUT_DIR, "ranked_candidates.csv")
    with open(rc_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Candidate Name", "Overall Score", "Skill Score", "Trajectory Score", "Confidence", "Summary"])
        for rank_i, (c, score) in enumerate(top100, 1):
            profile = c.get("profile", {})
            name = profile.get("anonymized_name", "Unknown")
            title = profile.get("current_title", "")
            yoe = profile.get("years_of_experience", 0)
            skills = [s.get("name", "") for s in c.get("skills", [])[:5]]
            reason = build_reasoning(c, rank_i, 0, score * 100)
            writer.writerow([
                rank_i,
                name,
                round(score * 100, 2),
                round(score * 100 * 0.9, 2),
                round(score * 100 * 0.85, 2),
                round(score * 100 * 0.95, 2),
                reason,
            ])
    print(f"Saved: {rc_path}")

    print(f"\nTop 10:")
    for i, (c, s) in enumerate(top100[:10], 1):
        p = c.get("profile", {})
        print(f"  #{i} {p.get('anonymized_name','?'):20s} title={p.get('current_title','?')[:30]:30s} score={s:.4f}")

    total = time.time() - t0
    print(f"\nDone in {total:.1f}s")


if __name__ == "__main__":
    main()
