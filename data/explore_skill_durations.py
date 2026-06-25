import json

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

stats = []
with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        skills = cand.get("skills", [])
        
        # count of skills with dur == 0 and proficiency in ['expert', 'advanced']
        expert_zero_dur = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
        adv_zero_dur = sum(1 for s in skills if s.get("proficiency") == "advanced" and s.get("duration_months", 0) == 0)
        any_zero_dur = sum(1 for s in skills if s.get("duration_months", 0) == 0)
        
        if expert_zero_dur > 0 or adv_zero_dur > 0 or any_zero_dur >= 5:
            stats.append({
                "cid": cand["candidate_id"],
                "expert_zero": expert_zero_dur,
                "adv_zero": adv_zero_dur,
                "any_zero": any_zero_dur
            })

print(f"Total candidates with zero dur skills: {len(stats)}")
# Print top 15 with highest expert_zero or any_zero
stats.sort(key=lambda x: x["expert_zero"] + x["adv_zero"], reverse=True)
for s in stats[:20]:
    print(s)
