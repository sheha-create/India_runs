import json
import os
from datetime import datetime

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

def detect_honeypot(cand):
    cid = cand["candidate_id"]
    profile = cand["profile"]
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    signals = cand.get("redrob_signals", {})
    
    # Rule 1: Salary inversion
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal and sal.get("min", 0) > sal.get("max", float("inf")):
        return True, "salary_inversion"
        
    # Rule 2: Career start vs College start
    # Full-time job started more than 5 years before college start (excluding interns)
    if edu:
        min_edu_start = min((e["start_year"] for e in edu), default=9999)
        for job in career:
            sd = job.get("start_date")
            title = job.get("title", "").lower()
            if sd and "intern" not in title and "trainee" not in title and "student" not in title:
                try:
                    start_year = datetime.strptime(sd, "%Y-%m-%d").year
                    if start_year < min_edu_start - 5:
                        return True, f"career_before_college_job_started_{start_year}_edu_{min_edu_start}"
                except Exception:
                    pass
                    
    # Rule 3: Single job duration massive discrepancy
    # E.g. dates say 12 months but duration_months claims 100 months, or vice versa
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
                    e_dt = datetime(2026, 6, 17) # current date of competition
                diff_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                if abs(diff_months - dur) > 24: # Allowing substantial leeway but flagging major errors
                    return True, f"job_duration_discrepancy_dates_{diff_months}_dur_{dur}"
            except Exception:
                pass
                
    # Rule 4: Stated YOE vs career history duration
    # Stated YOE is high but career history sum of durations is extremely low
    yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(job.get("duration_months", 0) for job in career)
    total_career_years = total_career_months / 12.0
    if yoe > 5.0 and total_career_years < 1.0:
        return True, f"high_yoe_{yoe}_low_career_dur_{total_career_years:.1f}"
    if total_career_years > yoe + 6.0:
        return True, f"career_dur_{total_career_years:.1f}_exceeds_yoe_{yoe}"

    # Rule 5: Invalid rates
    for rate_field in ["recruiter_response_rate", "interview_completion_rate"]:
        val = signals.get(rate_field, 0)
        if val is not None and (val < 0.0 or val > 1.0):
            return True, f"invalid_rate_{rate_field}_{val}"
            
    offer_rate = signals.get("offer_acceptance_rate", 0)
    if offer_rate is not None and (offer_rate < -1.0 or offer_rate > 1.0):
        return True, f"invalid_offer_rate_{offer_rate}"
        
    github_score = signals.get("github_activity_score", 0)
    if github_score is not None and (github_score < -1.0 or github_score > 100.0):
        return True, f"invalid_github_score_{github_score}"

    return False, ""

print("Scanning full candidates.jsonl...")
honeypots = []
total_count = 0
with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        is_hp, reason = detect_honeypot(cand)
        if is_hp:
            honeypots.append((cand["candidate_id"], reason))
        total_count += 1

print(f"Scanned {total_count} candidates.")
print(f"Found {len(honeypots)} honeypots.")
print("First 20 honeypots:")
for cid, reason in honeypots[:20]:
    print(f"  {cid}: {reason}")
