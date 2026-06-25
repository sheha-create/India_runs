import json
from collections import Counter

large_file = r"c:\Users\Home\redrob\data\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

from test_honeypots_rules import detect_honeypot

reasons = []
with open(large_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        cand = json.loads(line)
        is_hp, reason = detect_honeypot(cand)
        if is_hp:
            # Get the general category of reason
            cat = reason.split("_")[0]
            reasons.append(cat)

print(Counter(reasons))
