import json

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        if cand["candidate_id"] == "CAND_0001185":
            print(f"Education for {cand['candidate_id']}:")
            print(json.dumps(cand['education'], indent=2))
            print("Career History:")
            for job in cand['career_history']:
                print(f"  {job['company']} | {job['title']} | Start: {job['start_date']} | End: {job['end_date']}")
            break
