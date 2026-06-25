import json
import os
from datetime import datetime

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

mismatch_count = 0
expert_skills_count = 0
double_anom = 0

with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        cid = cand["candidate_id"]
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        
        # Check job duration mismatch
        has_job_mismatch = False
        for job in career:
            sd = job.get("start_date")
            ed = job.get("end_date")
            dur = job.get("duration_months", 0)
            if sd:
                try:
                    s_dt = datetime.strptime(sd, "%Y-%m-%d")
                    if ed:
                        e_dt = datetime.strptime(ed, "%Y-%m-%d")
                    else:
                        e_dt = datetime(2026, 6, 17)
                    diff_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                    # If duration of the job is much larger than dates difference (e.g. by 2 years or more)
                    if dur > diff_months + 12:
                        has_job_mismatch = True
                except Exception:
                    pass
        
        # Check expert/advanced skills with 0 duration
        expert_zero_dur = 0
        for s in skills:
            prof = s.get("proficiency", "")
            dur = s.get("duration_months", 0)
            if prof in ["expert", "advanced"] and dur == 0:
                expert_zero_dur += 1
                
        is_skill_honeypot = (expert_zero_dur >= 10)
        
        if has_job_mismatch:
            mismatch_count += 1
        if is_skill_honeypot:
            expert_skills_count += 1
        if has_job_mismatch or is_skill_honeypot:
            double_anom += 1

print(f"Total job duration mismatches: {mismatch_count}")
print(f"Total skill honeypots (>=10 expert/adv skills with 0 dur): {expert_skills_count}")
print(f"Total combined honeypots: {double_anom}")
