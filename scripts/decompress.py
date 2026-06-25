"""Post-install script: decompress candidates.jsonl.gz if needed."""
import gzip
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "[PUB] India_runs_data_and_ai_challenge" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"

GZ = DATA_DIR / "candidates.jsonl.gz"
JSONL = DATA_DIR / "candidates.jsonl"

if GZ.exists() and not JSONL.exists():
    print(f"Decompressing {GZ.name}...")
    with gzip.open(GZ, "rb") as f_in:
        with open(JSONL, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
    print(f"Done: {JSONL.stat().st_size / 1e6:.1f} MB")
elif JSONL.exists():
    print("candidates.jsonl already exists")
else:
    print("No candidates.jsonl.gz found")
