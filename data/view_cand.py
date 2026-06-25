import json

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

target_ids = ["CAND_0000009"]

with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        if cand["candidate_id"] in target_ids:
            print(f"=== {cand['candidate_id']} ===")
            print(json.dumps(cand["redrob_signals"]["expected_salary_range_inr_lpa"], indent=2))
            print()
