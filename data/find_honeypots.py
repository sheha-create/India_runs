import json
from datetime import datetime

with open(r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json", "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Loaded {len(candidates)} candidates.")

for cand in candidates:
    cid = cand["candidate_id"]
    profile = cand["profile"]
    career = cand["career_history"]
    skills = cand["skills"]
    edu = cand["education"]
    signals = cand["redrob_signals"]
    
    anomalies = []
    
    # Check YOE vs career history
    total_months = sum(job["duration_months"] for job in career)
    yoe = profile["years_of_experience"]
    if abs(total_months / 12 - yoe) > 2.0:
        anomalies.append(f"yoe discrepancy: profile says {yoe} but sum of career is {total_months/12:.2f} yrs")
        
    # Check skills with high proficiency but 0 duration
    for s in skills:
        name = s["name"]
        prof = s["proficiency"]
        dur = s.get("duration_months", 0)
        if prof in ["expert", "advanced"] and dur == 0:
            anomalies.append(f"skill anomaly: {name} is {prof} but duration_months is 0")
            
    # Check education dates vs career dates
    # If starting full-time career (excluding internships) way before college or during early college
    min_edu_start = min((e["start_year"] for e in edu), default=9999)
    for job in career:
        if job["start_date"]:
            try:
                start_year = datetime.strptime(job["start_date"], "%Y-%m-%d").year
                # If they started working full time before college start
                if start_year < min_edu_start - 4 and not "intern" in job["title"].lower():
                    anomalies.append(f"job start anomaly: started {job['company']} in {start_year} before edu start {min_edu_start}")
            except Exception:
                pass
                
    # Check date overlaps
    # Let's see if duration_months matches start_date/end_date difference
    for job in career:
        sd, ed = job["start_date"], job["end_date"]
        dur = job["duration_months"]
        if sd:
            try:
                s_dt = datetime.strptime(sd, "%Y-%m-%d")
                if ed:
                    e_dt = datetime.strptime(ed, "%Y-%m-%d")
                else:
                    e_dt = datetime(2026, 6, 17) # Current date from PRD
                diff_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                if abs(diff_months - dur) > 3:
                    anomalies.append(f"job duration anomaly at {job['company']}: dates say {diff_months} months, duration_months says {dur}")
            except Exception as e:
                pass

    if anomalies:
        print(f"Candidate {cid} ({profile['anonymized_name']}):")
        for a in anomalies:
            print(f"  - {a}")
