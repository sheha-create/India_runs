"""
app/rays_server.py — Rays AI FastAPI Backend
Run: python app/rays_server.py
Visit: http://127.0.0.1:8502
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import config

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticBase

app = FastAPI(title="Rays AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CANDIDATES_JSONL = (
    ROOT / "data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)
EMBEDDINGS_NPY = ROOT / "data" / "candidate_embeddings.npy"
IDS_JSON       = ROOT / "data" / "candidate_ids.json"
HTML_FILE      = Path(__file__).parent / "rays_ai.html"


class RankRequest(PydanticBase):
    jd_text: str
    top_n: int = 10
    groq_api_key: str = ""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    if not HTML_FILE.exists():
        raise HTTPException(404, "Frontend not built")
    return FileResponse(str(HTML_FILE), media_type="text/html")


@app.get("/api/status")
async def get_status():
    return {
        "candidates_available": CANDIDATES_JSONL.exists(),
        "embeddings_precomputed": EMBEDDINGS_NPY.exists() and IDS_JSON.exists(),
        "groq_configured": bool(config.GROQ_API_KEY),
        "total_candidates": 100000 if CANDIDATES_JSONL.exists() else 0,
        "groq_model": config.GROQ_MODEL,
    }


@app.post("/api/rank")
async def rank_candidates(req: RankRequest):
    if req.groq_api_key:
        config.GROQ_API_KEY = req.groq_api_key

    if not config.GROQ_API_KEY:
        raise HTTPException(400, "Groq API key not configured. Add GROQ_API_KEY to .env or pass it in the request.")

    if not CANDIDATES_JSONL.exists():
        raise HTTPException(404, "Candidates dataset not found at expected path.")

    try:
        import numpy as np
        from modules.jd_parser import JDParser
        from modules.candidate_profiler import CandidateProfiler, map_jsonl_to_flat_dict
        from modules.embedder import Embedder
        from modules.ranker import Ranker
        from rank import is_honeypot

        embedder = Embedder()
        jd_parser = JDParser()
        profiler = CandidateProfiler(use_llm=True)
        ranker = Ranker()

        role_profile = jd_parser.parse(req.jd_text)
        top_n = max(1, min(req.top_n, 50))
        embeddings_exist = EMBEDDINGS_NPY.exists() and IDS_JSON.exists()

        if embeddings_exist:
            cand_embeddings = np.load(str(EMBEDDINGS_NPY))
            with open(str(IDS_JSON)) as f:
                cand_ids = json.load(f)

            jd_emb = embedder.model.encode(
                [role_profile.to_embedding_text()], convert_to_numpy=True, show_progress_bar=False
            )
            cand_norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
            cand_norms[cand_norms == 0] = 1e-9
            cand_normed = cand_embeddings / cand_norms
            jd_norm = np.linalg.norm(jd_emb)
            jd_normed = jd_emb / (jd_norm if jd_norm > 0 else 1e-9)
            cos_sims = (cand_normed @ jd_normed.T).flatten()
            cos_sims_scaled = (cos_sims.clip(0, 1) * 100).tolist()
            ranked_pairs = sorted(zip(cand_ids, cos_sims_scaled), key=lambda x: x[1], reverse=True)
            target_ids = {p[0] for p in ranked_pairs[: top_n * 3]}

            raw_map: dict = {}
            with open(str(CANDIDATES_JSONL), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    cid = c.get("candidate_id")
                    if cid in target_ids:
                        raw_map[cid] = c
                    if len(raw_map) >= len(target_ids):
                        break

            valid_top = []
            for cid, emb_score in ranked_pairs:
                cd = raw_map.get(cid)
                if cd and not is_honeypot(cd):
                    valid_top.append((cd, emb_score))
                    if len(valid_top) >= top_n:
                        break

            candidate_profiles = []
            for cd, emb_score in valid_top:
                flat = map_jsonl_to_flat_dict(cd)
                profile = profiler.profile_row(flat)
                candidate_profiles.append((profile, emb_score))

        else:
            raw_cands: list = []
            with open(str(CANDIDATES_JSONL), encoding="utf-8") as f:
                for line in f:
                    if len(raw_cands) >= 200:
                        break
                    if line.strip():
                        raw_cands.append(json.loads(line))
            import pandas as pd
            flat_rows = [map_jsonl_to_flat_dict(c) for c in raw_cands]
            fp = CandidateProfiler(use_llm=False)
            profiles = fp.profile_dataframe(pd.DataFrame(flat_rows), show_progress=False)
            candidate_profiles = embedder.rank_candidates(role_profile, profiles)[:top_n]

        scored = ranker.rank(role_profile, candidate_profiles, show_progress=False)

        results = []
        for sc in scored:
            p = sc.profile
            all_skills = list(dict.fromkeys(p.explicit_skills + p.inferred_skills))
            results.append({
                "rank": sc.rank,
                "candidate_id": p.candidate_id,
                "name": p.name,
                "current_title": p.current_title,
                "years_experience": round(p.total_years_experience, 1),
                "skills": all_skills[:10],
                "overall_score": round(sc.overall_score, 1),
                "skill_match": round(sc.skill_match, 1),
                "experience_relevance": round(sc.experience_relevance, 1),
                "behavioral_fit": round(sc.behavioral_fit, 1),
                "platform_signal": round(sc.platform_signal, 1),
                "embedding_similarity": round(sc.embedding_similarity, 1),
                "rationale": sc.rationale,
                "skill_evidence": sc.skill_evidence[:5],
                "experience_evidence": sc.experience_evidence[:5],
                "education": p.education or "",
                "career_trajectory": p.career_trajectory or "",
                "domain_exposure": p.domain_exposure[:3],
                "behavioral_signals": p.behavioral_signals[:3],
            })

        return {
            "success": True,
            "role_profile": role_profile.model_dump(),
            "results": results,
            "total_ranked": len(results),
            "embeddings_used": embeddings_exist,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    print("\n\033[1;36m✦ Rays AI Platform\033[0m")
    print("━" * 40)
    print(f"  🌐  http://127.0.0.1:8502")
    print(f"  📡  API: http://127.0.0.1:8502/api/status")
    print("━" * 40 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8502, reload=False)
