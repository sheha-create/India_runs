import json, sys
sys.path.insert(0, '.')
from rank import _title_score, _skill_overlap, _platform_signal_score, _experience_score, REQUIRED_SKILLS, PREFERRED_SKILLS

sample_file = r'data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json'
with open(sample_file, 'r', encoding='utf-8') as f:
    cands = json.load(f)

check_ids = {'CAND_0000033', 'CAND_0000007', 'CAND_0000031'}
for c in cands:
    cid = c['candidate_id']
    if cid not in check_ids:
        continue
    prof = c['profile']
    skills = c.get('skills', [])
    signals = c.get('redrob_signals', {})
    title = prof.get('current_title', '')
    yoe = prof.get('years_of_experience', 0)
    print(f'=== {cid} | {title} ===')
    print(f'  YOE={yoe}')
    print(f'  title_score={_title_score(title):.3f}')
    print(f'  exp_score={_experience_score(yoe):.3f}')
    req = _skill_overlap(skills, REQUIRED_SKILLS)
    pref = _skill_overlap(skills, PREFERRED_SKILLS)
    print(f'  req_skill_overlap={req:.3f}, pref={pref:.3f}')
    matched_req = [s['name'] for s in skills if any(t in s['name'].lower() for t in REQUIRED_SKILLS)]
    print(f'  matched_req_skills: {matched_req}')
    print(f'  platform_score={_platform_signal_score(signals):.3f}')
    print()
