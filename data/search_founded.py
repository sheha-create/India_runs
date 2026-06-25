import json

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

found_count = 0
with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if "founded" in line.lower():
            cand = json.loads(line)
            print(f"Candidate {cand['candidate_id']}:")
            # print where 'founded' occurs
            for job in cand.get("career_history", []):
                desc = job.get("description", "")
                if "founded" in desc.lower():
                    print(f"  Job desc: {desc}")
                comp = job.get("company", "")
                if "founded" in comp.lower():
                    print(f"  Company: {comp}")
            found_count += 1
            if found_count >= 10:
                break
