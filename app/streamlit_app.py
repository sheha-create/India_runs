"""RedRob AI — Enterprise Candidate Intelligence Workspace."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure data directories exist
(ROOT / "data" / "output").mkdir(parents=True, exist_ok=True)

import streamlit as st

st.set_page_config(
    page_title="RedRob AI · Candidate Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config

# Lazy heavy imports — only loaded when the ranking pipeline actually runs
@st.cache_resource(show_spinner=False)
def _get_heavy_deps():
    """Import heavy dependencies once and cache them."""
    import numpy as np
    import pandas as pd
    from modules.candidate_profiler import CandidateProfiler, map_jsonl_to_flat_dict
    from modules.embedder import Embedder
    from modules.jd_parser import JDParser
    from modules.output_writer import _to_record
    from modules.ranker import Ranker
    from schemas.candidate_profile import CandidateProfile, ScoredCandidate
    return {
        "np": np, "pd": pd,
        "CandidateProfiler": CandidateProfiler,
        "map_jsonl_to_flat_dict": map_jsonl_to_flat_dict,
        "Embedder": Embedder,
        "JDParser": JDParser,
        "_to_record": _to_record,
        "Ranker": Ranker,
        "CandidateProfile": CandidateProfile,
        "ScoredCandidate": ScoredCandidate,
    }

# Paths
CANDIDATES_JSONL = ROOT / "data" / "[PUB] India_runs_data_and_ai_challenge" / \
    "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
EMBEDDINGS_NPY = ROOT / "data" / "candidate_embeddings.npy"
IDS_JSON = ROOT / "data" / "candidate_ids.json"
DEMO_RESULTS = ROOT / "data" / "output" / "ranked_candidates.json"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a1f35;
            --bg-card-hover: #1f2847;
            --border: #2d3654;
            --border-glow: #3b82f6;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
        }
        
        .stApp {
            background: var(--bg-primary) !important;
            background-image: radial-gradient(at 40% 20%, hsla(228, 100%, 74%, 0.15) 0px, transparent 50%),
                             radial-gradient(at 80% 0%, hsla(189, 100%, 56%, 0.1) 0px, transparent 50%),
                             radial-gradient(at 0% 50%, hsla(355, 100%, 93%, 0.05) 0px, transparent 50%) !important;
            color: var(--text-primary) !important;
        }
        
        .stApp > header { background: transparent !important; }
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-primary) !important;
        }
        
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary) !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 12px 16px !important;
            border-radius: 10px !important;
            color: var(--text-secondary) !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(59, 130, 246, 0.1) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        }
        
        .card {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            transition: all 0.3s ease !important;
        }
        .card:hover {
            border-color: var(--border-glow) !important;
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.2) !important;
        }
        
        .metric-card {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
        }
        .metric-card::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important; left: 0 !important; right: 0 !important;
            height: 3px !important;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        }
        .metric-card:hover {
            border-color: var(--border-glow) !important;
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.2) !important;
        }
        .metric-label {
            font-size: 12px !important; font-weight: 600 !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !important; letter-spacing: 1px !important;
            margin-bottom: 8px !important;
        }
        .metric-value {
            font-size: 32px !important; font-weight: 800 !important;
            color: var(--text-primary) !important; margin-bottom: 8px !important;
        }
        .metric-change { font-size: 12px !important; font-weight: 600 !important; }
        .metric-change.positive { color: var(--accent-green) !important; }
        
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
            color: white !important; border: none !important;
            border-radius: 10px !important; font-weight: 600 !important;
            padding: 12px 24px !important; transition: all 0.2s ease !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }
        .stDownloadButton > button {
            background: var(--bg-card) !important; color: var(--text-primary) !important;
            border: 1px solid var(--border) !important; border-radius: 10px !important;
            font-weight: 600 !important;
        }
        
        .stTextInput input, .stTextArea textarea, .stNumberInput input,
        .stSelectbox > div > div {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }
        
        div[data-testid="stForm"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important; padding: 24px !important;
        }
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #3b82f6, #4f46e5) !important;
            color: white !important; border: none !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.2) !important;
        }
        
        .candidate-card {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important; padding: 20px !important;
            margin-bottom: 16px !important; transition: all 0.3s ease !important;
        }
        .candidate-card:hover {
            border-color: var(--border-glow) !important;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.15) !important;
            transform: translateY(-2px) !important;
        }
        .candidate-header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
        .candidate-avatar {
            width: 56px; height: 56px; border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 18px; color: white;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6); flex-shrink: 0;
        }
        .candidate-info { flex: 1; }
        .candidate-name { font-size: 18px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
        .candidate-title { font-size: 13px; color: var(--text-secondary); }
        .candidate-score { text-align: right; }
        .candidate-score .score-value {
            font-size: 28px; font-weight: 800;
            background: linear-gradient(135deg, #06b6d4, #3b82f6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .candidate-score .score-label { font-size: 10px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
        
        .skill-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
        .skill-tag {
            font-size: 11px; font-weight: 600; padding: 6px 12px;
            background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border-radius: 8px;
        }
        .score-bar { height: 8px; background: var(--bg-secondary); border-radius: 4px; overflow: hidden; margin-bottom: 16px; }
        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6);
            border-radius: 4px; transition: width 0.5s ease;
        }
        .rationale-box {
            background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: 12px; padding: 16px;
            border-left: 4px solid var(--accent-blue);
        }
        .rationale-title { font-size: 12px; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .rationale-text { font-size: 13px; color: var(--text-secondary); line-height: 1.6; }
        
        .section-header { margin-bottom: 24px; }
        .section-title { font-size: 20px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
        .section-subtitle { font-size: 14px; color: var(--text-secondary); }
        
        .status-pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 16px; border-radius: 20px;
            font-size: 12px; font-weight: 600;
        }
        .status-pill.online { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }
        
        .hero-section {
            background: linear-gradient(135deg, #0f172a, #1e293b 50%, #0f172a);
            border-radius: 20px; padding: 40px; margin-bottom: 32px;
            position: relative; overflow: hidden;
            border: 1px solid var(--border);
        }
        .hero-section::before {
            content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        }
        .hero-content { position: relative; z-index: 1; }
        .hero-title {
            font-size: 32px; font-weight: 800;
            background: linear-gradient(135deg, #f1f5f9, #94a3b8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 12px;
        }
        .hero-subtitle { font-size: 16px; color: var(--text-secondary); max-width: 600px; }
        
        .insight-card {
            background: linear-gradient(135deg, var(--from, #3b82f6), var(--to, #8b5cf6));
            border-radius: 16px; padding: 24px; color: white;
            min-height: 160px; position: relative; overflow: hidden;
        }
        .insight-card::before {
            content: ""; position: absolute; top: 0; right: 0;
            width: 120px; height: 120px;
            background: rgba(255, 255, 255, 0.1); border-radius: 50%;
            transform: translate(30%, -30%);
        }
        .insight-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; opacity: 0.9; }
        .insight-value { font-size: 32px; font-weight: 800; margin-bottom: 8px; }
        .insight-description { font-size: 12px; opacity: 0.8; line-height: 1.5; }
        
        .funnel-item { display: flex; align-items: center; gap: 16px; padding: 12px 0; border-bottom: 1px solid var(--border); }
        .funnel-item:last-child { border-bottom: none; }
        .funnel-label { width: 100px; font-size: 13px; font-weight: 600; color: var(--text-secondary); }
        .funnel-bar { flex: 1; height: 8px; background: var(--bg-secondary); border-radius: 4px; overflow: hidden; }
        .funnel-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 4px; }
        .funnel-value { width: 80px; text-align: right; font-size: 14px; font-weight: 700; color: var(--text-primary); }
        
        .activity-item { display: flex; align-items: center; gap: 16px; padding: 16px 0; border-bottom: 1px solid var(--border); }
        .activity-item:last-child { border-bottom: none; }
        .activity-avatar {
            width: 44px; height: 44px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
            color: var(--accent-blue); flex-shrink: 0;
        }
        .activity-content { flex: 1; }
        .activity-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
        .activity-time { font-size: 12px; color: var(--text-muted); }
        
        .copilot-button {
            position: fixed; right: 32px; bottom: 32px;
            width: 60px; height: 60px; border-radius: 16px;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white; font-size: 24px; z-index: 1000;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        }
        
        .brand { display: flex; align-items: center; gap: 12px; padding: 0 16px 24px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
        .brand-icon {
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border-radius: 12px; display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 20px; color: white;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }
        .brand-text { font-size: 20px; font-weight: 700; color: var(--text-primary); }
        .brand-text span { color: var(--accent-purple); }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, eyebrow: str = "Workspace") -> None:
    st.markdown(
        f"""<div class="section-header">
        <div class="status-pill online">● {eyebrow}</div>
        <div class="section-title" style="margin-top: 12px;">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def metric(label: str, value: str, delta: str, note: str, wash: str) -> None:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-change positive">↗ {delta} <span style="color: var(--text-muted); font-weight: 500; margin-left: 4px;">{note}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


def load_demo_results() -> list:
    """Load demo results if available. Returns list of ScoredCandidate or empty list."""
    if not DEMO_RESULTS.exists():
        return []
    try:
        deps = _get_heavy_deps()
        ScoredCandidate = deps["ScoredCandidate"]
        CandidateProfile = deps["CandidateProfile"]
        with DEMO_RESULTS.open(encoding="utf-8") as file:
            rows = json.load(file)
        results = []
        for row in rows:
            scores = row.get("scores", {})
            profile = CandidateProfile(
                candidate_id=row.get("candidate_id", ""),
                name=row.get("name", "Candidate"),
                email=row.get("email", ""),
                current_title=row.get("current_title", ""),
                total_years_experience=row.get("total_years_experience", 0),
                explicit_skills=row.get("skills", {}).get("explicit", []),
                inferred_skills=row.get("skills", {}).get("inferred", []),
                education=row.get("education", ""),
                career_trajectory=row.get("career_trajectory", ""),
                domain_exposure=row.get("domain_exposure", []),
                behavioral_signals=row.get("behavioral_signals", []),
                platform_signals=row.get("platform_signals", []),
            )
            results.append(ScoredCandidate(
                profile=profile, rank=row.get("rank", len(results) + 1),
                overall_score=scores.get("overall", 0), skill_match=scores.get("skill_match", 0),
                experience_relevance=scores.get("experience_relevance", 0),
                behavioral_fit=scores.get("behavioral_fit", 0),
                platform_signal=scores.get("platform_signal", 0),
                embedding_similarity=scores.get("embedding_similarity", 0),
                rationale=row.get("rationale", ""), skill_evidence=row.get("skill_evidence", []),
                experience_evidence=row.get("experience_evidence", []),
            ))
        return results
    except Exception:
        return []


def run_pipeline(jd_text: str, top_n: int) -> list:
    deps = _get_heavy_deps()
    np = deps["np"]
    pd = deps["pd"]
    CandidateProfiler = deps["CandidateProfiler"]
    map_jsonl_to_flat_dict = deps["map_jsonl_to_flat_dict"]
    Embedder = deps["Embedder"]
    JDParser = deps["JDParser"]
    Ranker = deps["Ranker"]
    ScoredCandidate = deps["ScoredCandidate"]

    role_profile = JDParser().parse(jd_text)
    profiler = CandidateProfiler(use_llm=True)
    embedder = Embedder()
    top_candidates = []
    remaining_candidates = []

    if EMBEDDINGS_NPY.exists() and IDS_JSON.exists():
        embeddings = np.load(EMBEDDINGS_NPY)
        with IDS_JSON.open() as file:
            candidate_ids = json.load(file)
        jd_embedding = embedder.model.encode(
            [role_profile.to_embedding_text()], convert_to_numpy=True, show_progress_bar=False
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1e-9, norms)
        jd_normalized = jd_embedding / max(np.linalg.norm(jd_embedding), 1e-9)
        similarities = (normalized @ jd_normalized.T).flatten().clip(0, 1) * 100
        ranked_pairs = sorted(zip(candidate_ids, similarities.tolist()), key=lambda item: item[1], reverse=True)
        target_ids = {candidate_id for candidate_id, _ in ranked_pairs[: top_n * 3]}
        raw_map = {}
        with CANDIDATES_JSONL.open(encoding="utf-8") as file:
            for line in file:
                candidate = json.loads(line)
                if candidate.get("candidate_id") in target_ids:
                    raw_map[candidate["candidate_id"]] = candidate
        from rank import is_honeypot
        for candidate_id, score in ranked_pairs:
            candidate = raw_map.get(candidate_id)
            if candidate and not is_honeypot(candidate):
                top_candidates.append((profiler.profile_row(map_jsonl_to_flat_dict(candidate)), score))
                if len(top_candidates) >= top_n:
                    break
    else:
        if not CANDIDATES_JSONL.exists():
            raise FileNotFoundError("Candidate database not found. Please upload candidates.jsonl to the data directory.")
        raw_candidates = []
        with CANDIDATES_JSONL.open(encoding="utf-8") as file:
            for line in file:
                if len(raw_candidates) >= 200:
                    break
                raw_candidates.append(json.loads(line))
        fallback = CandidateProfiler(use_llm=False)
        profiles = fallback.profile_dataframe(
            pd.DataFrame([map_jsonl_to_flat_dict(candidate) for candidate in raw_candidates]),
            show_progress=False,
        )
        ranked = embedder.rank_candidates(role_profile, profiles)
        top_candidates, remaining_candidates = ranked[:top_n], ranked[top_n:]

    scored = list(Ranker().rank(role_profile, top_candidates, show_progress=False))
    for profile, score in remaining_candidates:
        scored.append(ScoredCandidate(
            profile=profile, rank=len(scored) + 1, overall_score=score, skill_match=score,
            experience_relevance=score, behavioral_fit=50, platform_signal=50,
            embedding_similarity=score, rationale=f"Semantic match score: {score:.1f}.",
        ))
    st.session_state.role_profile = role_profile
    return scored


def dashboard() -> None:
    page_header("Security Operations Center", "Real-time threat intelligence and candidate monitoring.", "Threat Intelligence")
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-title">Candidate Intelligence Platform</div>
            <div class="hero-subtitle">AI-powered talent analysis with real-time threat detection and behavioral analytics. Monitor candidate signals and identify high-value opportunities.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    values = [
        ("Total Candidates", "100,000", "Active", "Real-time monitoring", "#1a1f35"),
        ("AI Engine", "Online", "100%", "System operational", "#1a1f35"),
        ("Match Accuracy", "94.2%", "+2.1%", "ML precision rate", "#1a1f35"),
        ("Threat Score", "Low", "0.3%", "Risk assessment", "#1a1f35"),
    ]
    for col, args in zip(cols, values):
        with col:
            metric(*args)
    
    left, right = st.columns([1.6, 1])
    
    with left:
        st.markdown("""<div class="card"><div class="section-title">Talent Acquisition Funnel</div>
        <div class="section-subtitle">Candidate movement across all active roles</div></div>""", unsafe_allow_html=True)
        
        for label, value, pct in [("Sourced", "100,000", 100), ("Screened", "45,200", 72),
                                   ("Shortlisted", "12,800", 49), ("Interview", "3,420", 31), ("Offers", "892", 15)]:
            st.markdown(f"""<div class="funnel-item"><div class="funnel-label">{label}</div>
            <div class="funnel-bar"><div class="funnel-fill" style="width:{pct}%"></div></div>
            <div class="funnel-value">{value}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="card" style="margin-top: 24px;"><div class="section-title">Threat Vector Analysis</div>
        <div class="section-subtitle">Most critical skill gaps in your pipeline</div></div>""", unsafe_allow_html=True)
        
        for label, pct, severity in [("Python", 92, "Critical"), ("Machine Learning", 78, "High"),
                                      ("NLP/IR", 65, "High"), ("System Design", 54, "Medium"), ("MLOps", 42, "Medium")]:
            color = "#ef4444" if severity == "Critical" else "#f59e0b" if severity == "High" else "#3b82f6"
            st.markdown(f"""<div class="funnel-item"><div class="funnel-label">{label}</div>
            <div class="funnel-bar"><div class="funnel-fill" style="width:{pct}%; background:{color}"></div></div>
            <div class="funnel-value" style="color:{color}">{pct}%</div></div>""", unsafe_allow_html=True)
    
    with right:
        st.markdown("""<div class="card"><div class="section-title">Real-time Activity Feed</div>
        <div class="section-subtitle">Live updates from candidate monitoring</div></div>""", unsafe_allow_html=True)
        
        for initials, name, action, when, status in [
            ("SI", "System Intelligence", "New candidate match detected", "2 min ago", "online"),
            ("AI", "AI Engine", "Ranking model updated", "15 min ago", "online"),
            ("TM", "Team Lead", "Shortlisted 12 candidates", "1 hr ago", "online"),
            ("HM", "Hiring Manager", "Created new requisition", "2 hr ago", "warning"),
            ("RC", "Recruiter", "Moved candidate to interview", "3 hr ago", "online"),
        ]:
            sc = "#10b981" if status == "online" else "#f59e0b"
            st.markdown(f"""<div class="activity-item"><div class="activity-avatar">{initials}</div>
            <div class="activity-content"><div class="activity-title">{name}</div>
            <div class="activity-time">{action} · {when}</div></div>
            <div style="width:8px;height:8px;border-radius:50%;background:{sc};"></div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="card" style="margin-top: 24px;"><div class="section-title">System Health</div>
        <div class="section-subtitle">Infrastructure status monitoring</div></div>""", unsafe_allow_html=True)
        
        for name, status, pct, color in [
            ("API Gateway", "Operational", 100, "#10b981"),
            ("ML Pipeline", "Operational", 100, "#10b981"),
            ("Database", "Operational", 100, "#10b981"),
            ("Cache Layer", "Degraded", 87, "#f59e0b"),
        ]:
            st.markdown(f"""<div class="funnel-item"><div class="funnel-label">{name}</div>
            <div class="funnel-bar"><div class="funnel-fill" style="width:{pct}%; background:{color}"></div></div>
            <div class="funnel-value" style="color:{color}">{status}</div></div>""", unsafe_allow_html=True)


def render_candidate(candidate) -> None:
    profile = candidate.profile
    initials = "".join(part[0] for part in profile.name.split()[:2]).upper()
    skills = (candidate.skill_evidence or profile.explicit_skills)[:6]
    skill_tags = "".join(f'<span class="skill-tag">{html.escape(skill)}</span>' for skill in skills)
    rationale = html.escape(candidate.rationale or "Strong alignment across the role's core requirements.")
    
    st.markdown(
        f"""<div class="candidate-card">
        <div class="candidate-header">
            <div class="candidate-avatar">{initials}</div>
            <div class="candidate-info">
                <div class="candidate-name">#{candidate.rank} · {html.escape(profile.name)}</div>
                <div class="candidate-title">{html.escape(profile.current_title or "Candidate")} · {profile.total_years_experience:.1f} years experience</div>
            </div>
            <div class="candidate-score">
                <div class="score-value">{candidate.overall_score:.0f}%</div>
                <div class="score-label">AI MATCH</div>
            </div>
        </div>
        <div class="skill-tags">{skill_tags}</div>
        <div class="score-bar"><div class="score-bar-fill" style="width:{min(candidate.overall_score, 100)}%"></div></div>
        <div class="rationale-box"><div class="rationale-title">✦ Threat Analysis</div>
        <div class="rationale-text">{rationale}</div></div></div>""",
        unsafe_allow_html=True,
    )


def ranking_workspace() -> None:
    page_header("Threat Analysis Workspace", "Configure attack vectors and let RedRob identify high-value targets.", "Intelligence Gathering")
    
    st.markdown("""<div class="hero-section"><div class="hero-content">
        <div class="status-pill online">● AI Engine Online</div>
        <div class="hero-title" style="margin-top: 16px;">Precision Targeting System</div>
        <div class="hero-subtitle">Advanced semantic analysis with explainable AI reasoning. Identify, evaluate, and prioritize high-value candidates with military-grade precision.</div>
    </div></div>""", unsafe_allow_html=True)
    
    builder, results_col = st.columns([0.82, 1.18], gap="large")
    
    with builder:
        st.markdown("""<div class="card"><div class="section-title">Target Configuration</div>
        <div class="section-subtitle">Define engagement parameters for intelligent candidate analysis</div></div>""", unsafe_allow_html=True)
        
        with st.form("role_builder"):
            title = st.text_input("Target Designation", value="Senior Machine Learning Engineer")
            exp_col, location_col = st.columns(2)
            with exp_col:
                experience = st.number_input("Minimum Experience", 0, 30, 5)
            with location_col:
                location = st.text_input("Target Location", value="Bengaluru · Hybrid")
            skills = st.text_input("Required Capabilities", value="Python, PyTorch, NLP, AWS, Docker")
            education = st.text_input("Educational Background", value="Computer Science or equivalent")
            certifications = st.text_input("Certifications", placeholder="AWS ML Specialty, optional")
            salary = st.text_input("Compensation Range", value="₹35L – ₹55L")
            requirements = st.text_area("Mission Parameters",
                value="Own production ML systems, mentor engineers, and partner with product teams.", height=110)
            top_n = st.slider("Deep Analysis Targets", 5, 50, 20, 5)
            submit = st.form_submit_button("✦ Execute Analysis", use_container_width=True)
        
        jd_text = (
            f"We are hiring a {title} in {location}. Minimum {experience} years of experience. "
            f"Required skills: {skills}. Education: {education}. Certifications: {certifications or 'not required'}. "
            f"Compensation: {salary}. Additional expectations: {requirements}"
        )
        
        if submit:
            if not CANDIDATES_JSONL.exists():
                st.error("Target database not found. Upload candidates.jsonl to the data directory.")
            elif not config.GROQ_API_KEY:
                st.error("API credentials required. Set GROQ_API_KEY in your .env or environment variables.")
            else:
                with st.status("RedRob is scanning intelligence database...", expanded=True) as status:
                    st.write("Establishing secure connection")
                    st.write("Analyzing target parameters")
                    st.write("Running AI threat assessment")
                    try:
                        st.session_state.results = run_pipeline(jd_text, top_n)
                        status.update(label="Analysis Complete", state="complete", expanded=False)
                    except Exception as exc:
                        status.update(label="Analysis Failed", state="error")
                        st.error(f"Error: {str(exc)}")
        
        with st.expander("Advanced Configuration"):
            st.caption("Scoring weights are normalized automatically by the ranking engine.")
            st.slider("Skill Match Weight", 0, 100, 35)
            st.slider("Experience Relevance", 0, 100, 25)
            st.slider("Behavioral Analysis", 0, 100, 15)
            st.slider("Platform Signals", 0, 100, 10)
    
    with results_col:
        results = st.session_state.get("results") or load_demo_results()
        top_line = f"{len(results)} candidates ranked" if results else "Awaiting target parameters"
        
        st.markdown(f"""<div class="section-header"><div class="section-title">Live Threat Analysis</div>
        <div class="section-subtitle">{top_line} · Ordered by explainable composite score</div></div>""", unsafe_allow_html=True)
        
        if results:
            deps = _get_heavy_deps()
            pd = deps["pd"]
            _to_record = deps["_to_record"]
            
            controls = st.columns([1, 1, 1])
            with controls[0]:
                st.selectbox("Filter", ["All candidates", "90%+ match", "Shortlisted"], label_visibility="collapsed")
            with controls[1]:
                st.selectbox("Sort", ["Best match", "Experience", "Skill score"], label_visibility="collapsed")
            with controls[2]:
                records = pd.DataFrame([_to_record(candidate) for candidate in results])
                st.download_button("Export CSV", records.to_csv(index=False), "redrob_threat_analysis.csv", "text/csv", use_container_width=True)
            
            for candidate in results[:10]:
                render_candidate(candidate)
                actions = st.columns([1, 1, 1])
                actions[0].button("View Profile", key=f"profile_{candidate.profile.candidate_id}", use_container_width=True)
                actions[1].button("Compare", key=f"compare_{candidate.profile.candidate_id}", use_container_width=True)
                actions[2].button("Shortlist", key=f"shortlist_{candidate.profile.candidate_id}", type="primary", use_container_width=True)
        else:
            st.info("Complete target configuration to generate intelligence report.")


def insights() -> None:
    page_header("Predictive Analytics", "Strategic intelligence across talent market and pipeline.", "Market Intelligence")
    
    cols = st.columns(4)
    cards = [
        ("Skill Gap Analysis", "3 Critical", "Kubernetes, MLOps and system design are underrepresented.", "#3b82f6,#8b5cf6"),
        ("Hiring Risk Score", "Low · 18", "Pipeline quality is healthy across priority roles.", "#8b5cf6,#a855f7"),
        ("Talent Availability", "2,840", "Qualified profiles in your preferred markets.", "#06b6d4,#22d3ee"),
        ("Success Prediction", "86%", "Expected 12-month retention for current shortlist.", "#10b981,#34d399"),
    ]
    for col, (title, value, copy, gradient) in zip(cols, cards):
        with col:
            st.markdown(f"""<div class="insight-card" style="--from:{gradient.split(',')[0]};--to:{gradient.split(',')[1]}">
            <div class="insight-title">{title}</div><div class="insight-value">{value}</div>
            <div class="insight-description">{copy}</div></div>""", unsafe_allow_html=True)
    
    left, right = st.columns([1.35, 0.65])
    with left:
        st.markdown("""<div class="card"><div class="section-title">Market Threat Landscape</div>
        <div class="section-subtitle">Supply index by high-priority capability</div></div>""", unsafe_allow_html=True)
        try:
            pd = _get_heavy_deps()["pd"]
            chart_data = pd.DataFrame({
                "Qualified Talent": [42, 52, 57, 68, 73, 81, 88],
                "Open Demand": [38, 44, 53, 59, 67, 74, 79],
            }, index=["Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"])
            st.area_chart(chart_data, color=["#3b82f6", "#8b5cf6"], height=320)
        except Exception:
            st.info("Chart unavailable.")
    
    with right:
        st.markdown("""<div class="card"><div class="section-title">Compensation Intelligence</div>
        <div class="section-subtitle">Senior ML Engineer · Bengaluru</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="padding:24px;">
        <div class="metric-value" style="font-size:36px;">₹44.8L</div>
        <div class="metric-change positive" style="margin-bottom:24px;">↗ 6.2% <span style="color:var(--text-muted);font-weight:500;margin-left:4px;">market median YoY</span></div>
        </div>""", unsafe_allow_html=True)
        for label, pct in [("P25 · ₹36L", 48), ("Median · ₹45L", 68), ("P75 · ₹57L", 87)]:
            st.markdown(f"""<div class="funnel-item"><div class="funnel-label">{label}</div>
            <div class="funnel-bar"><div class="funnel-fill" style="width:{pct}%"></div></div></div>""", unsafe_allow_html=True)


def placeholder_page(name: str) -> None:
    page_header(name, "Secure workspace for your intelligence operations.", "RedRob Workspace")
    st.markdown(f"""<div class="hero-section"><div class="hero-content">
        <div class="status-pill online">● Connected</div>
        <div class="hero-title" style="margin-top:16px;">{name}</div>
        <div class="hero-subtitle">Collaborate, automate workflows, and keep every decision in one auditable system.</div>
    </div></div>""", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (title, copy) in zip(cols, [
        ("Secure Workflows", "Keep notes, decisions, owners, and next steps synchronized with end-to-end encryption."),
        ("Enterprise Controls", "Role-based access, audit history, and secure integrations with zero-trust architecture."),
        ("AI-Assisted Actions", "Let RedRob handle repetitive work while operators stay in control of critical decisions."),
    ]):
        with col:
            st.markdown(f"""<div class="card"><div class="section-title">{title}</div>
            <div class="section-subtitle">{copy}</div></div>""", unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────────────────
inject_styles()

with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-icon">R</div>
    <div class="brand-text">RedRob <span>AI</span></div></div>""", unsafe_allow_html=True)
    
    navigation = st.radio("Navigation",
        ["Dashboard", "Candidate Ranking", "Job Requisitions", "Talent Pipeline", "AI Insights", "Reports", "Team Workspace", "Settings"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    
    st.markdown("""<div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;margin-top:16px;">
        <div style="font-size:12px;font-weight:700;color:var(--text-primary);margin-bottom:4px;">Security Operations</div>
        <div style="font-size:11px;color:var(--text-muted);">Candidate Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)
    
    with st.expander("AI Engine Configuration"):
        api_key = st.text_input("Groq API Key", type="password", placeholder="Loaded from environment")
        if api_key:
            config.GROQ_API_KEY = api_key
        config.GROQ_MODEL = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"])
    
    status_color = "#10b981" if config.GROQ_API_KEY else "#f59e0b"
    status_text = "System Online" if config.GROQ_API_KEY else "No API Key"
    st.markdown(f"""<div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:12px;margin-top:16px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:8px;height:8px;border-radius:50%;background:{status_color};"></div>
            <span style="font-size:12px;font-weight:600;color:{status_color};">{status_text}</span>
        </div>
        <div style="font-size:10px;color:var(--text-muted);">Groq API · {config.GROQ_MODEL}</div>
    </div>""", unsafe_allow_html=True)

if navigation == "Dashboard":
    dashboard()
elif navigation == "Candidate Ranking":
    ranking_workspace()
elif navigation == "AI Insights":
    insights()
else:
    placeholder_page(navigation)

st.markdown('<div class="copilot-button" title="Ask RedRob Copilot">🛡️</div>', unsafe_allow_html=True)
