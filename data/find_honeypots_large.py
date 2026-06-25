import json
import os
from datetime import datetime

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

print(f"Scanning candidates.jsonl...")
if not os.path.exists(large_file):
    print("Error: candidates.jsonl not found.")
    exit(1)

count = 0
found_anomalies = 0

with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        cid = cand["candidate_id"]
        profile = cand["profile"]
        career = cand["career_history"]
        skills = cand["skills"]
        edu = cand["education"]
        signals = cand["redrob_signals"]
        
        anomalies = []
        
        # 1. Skill duration anomalies: expert/advanced skill with 0 duration_months
        expert_zero_dur = 0
        for s in skills:
            prof = s["proficiency"]
            dur = s.get("duration_months", 0)
            if prof in ["expert", "advanced"] and dur == 0:
                expert_zero_dur += 1
        if expert_zero_dur >= 5:
            anomalies.append(f"expert/advanced skills with 0 duration count: {expert_zero_dur}")

        # 2. Total duration of career history significantly exceeding total YOE
        total_career_months = sum(job["duration_months"] for job in career)
        yoe = profile["years_of_experience"]
        if total_career_months / 12 > yoe + 5.0:
            anomalies.append(f"total career duration ({total_career_months/12:.1f} yrs) exceeds profile YOE ({yoe} yrs)")

        # 3. Impossible dates (e.g. started working at a company before it was founded or impossible durations)
        # Check if duration_months matches the difference between start_date and end_date
        for job in career:
            sd, ed = job["start_date"], job["end_date"]
            dur = job["duration_months"]
            if sd:
                try:
                    s_dt = datetime.strptime(sd, "%Y-%m-%d")
                    if ed:
                        e_dt = datetime.strptime(ed, "%Y-%m-%d")
                    else:
                        e_dt = datetime(2026, 6, 17) # Current date
                    diff_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                    # If duration_months is impossible (e.g., says 96 months (8 years) but dates cover only 12 months)
                    if abs(diff_months - dur) > 24: # Allowing some leeway but flagging massive differences
                        anomalies.append(f"job duration anomaly at {job['company']}: dates cover {diff_months} months, duration_months says {dur}")
                except Exception:
                    pass

        # 4. Starting full time career years before college start or birth
        if edu:
            min_edu_start = min((e["start_year"] for e in edu), default=9999)
            for job in career:
                if job["start_date"]:
                    try:
                        start_year = datetime.strptime(job["start_date"], "%Y-%m-%d").year
                        if start_year < min_edu_start - 6 and not "intern" in job["title"].lower():
                            anomalies.append(f"started career ({start_year}) long before college start ({min_edu_start})")
                    except Exception:
                        pass

        if anomalies:
            print(f"Honeypot candidate {cid} ({profile['anonymized_name']}):")
            for a in anomalies:
                print(f"  - {a}")
            found_anomalies += 1
            if found_anomalies >= 15:
                break
        
        count += 1
        if count >= 20000: # Scan first 20,000 rows
            break

print(f"Scanned {count} rows. Found {found_anomalies} anomalies.")
