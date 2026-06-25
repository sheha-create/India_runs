# 🎯 AI Candidate Ranking System

A robust, hybrid AI-powered recruiter agent that parses Job Descriptions, structures candidates' raw information, performs fast semantic filtering, and uses an LLM-as-a-judge reasoning step to rank and produce a trustworthy shortlist with detailed rationales.

---

## 🚀 Key Features
1. **JD Understanding Module**: Parses raw unstructured text into structured requirement profile JSON using LLMs.
2. **Candidate Profiling Module**: Cleans and structures raw candidate files, extracting inferred skills, career trajectory, behavioral indicators, and platform signals.
3. **Embedding Pre-Filter Layer**: Fast semantic search using local Sentence-Transformers (`all-MiniLM-L6-v2`) to filter down candidates.
4. **LLM-as-a-Judge Pass**: Performs structured evaluation on top candidates across 4 dimensions: Skill Match, Experience Relevance, Behavioral Fit, and Platform Signals.
5. **Weighted Scoring & Output**: Combines dimensions into a single normalized score and outputs results as CSV and JSON with complete transparency/rationale.
6. **Premium Web Dashboard**: A built-in Streamlit app to run pipelines, adjust weights, visual progress, and inspect candidates.

---

## 🛠️ Tech Stack
* **Language**: Python 3.8+
* **Libraries**: `sentence-transformers`, `pandas`, `pydantic`, `google-generativeai` (Gemini API), `openai` (optional), `streamlit`, `rich`
* **Models**: `gemini-1.5-flash` / `gpt-4o-mini` (LLM Reasoning), `all-MiniLM-L6-v2` (Embeddings)

---

## 📦 Setup & Installation

1. **Clone/Navigate to the directory**:
   ```bash
   cd redrob
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   * Copy the environment file template:
     ```bash
     cp .env.example .env
     ```
   * Open `.env` and fill in your Gemini API key (or OpenAI key if preferred):
     ```env
     LLM_PROVIDER=gemini
     GEMINI_API_KEY=AIzaSy...
     ```

---

## 👥 Expected Input Data Format

The candidate CSV file should have some of the following columns (missing columns will be automatically parsed with default/neutral values):

| Column Name | Description |
|---|---|
| `candidate_id` | Unique ID of the candidate (optional) |
| `name` | Full Name of the candidate (Required) |
| `current_title` | Current job title (e.g. Senior Software Engineer) |
| `years_experience` | Numerical years of experience |
| `skills` | Comma/semicolon separated list of explicit skills |
| `summary` | Free-text bio/summary |
| `work_history` | Free-text summary of past roles |
| `projects` | Free-text list of projects or descriptions |
| `behavioral_notes` | Optional communication style or behavioral notes |
| `github_url` | Candidate's GitHub URL |
| `linkedin_url` | Candidate's LinkedIn URL |
| `education` | Education background |

---

## 🚀 How to Run

### 1. Command Line Interface (CLI)
To run the end-to-end pipeline via the command line:
```bash
python main.py --jd data/sample_jd.txt --candidates path/to/your/candidates.csv --top-n 10
```

**Options**:
* `--jd`: Path to the job description file.
* `--candidates`: Path to the candidate CSV file.
* `--top-n`: Number of candidates passed to the LLM reasoning step (defaults to 20).
* `--no-llm-profile`: Disables candidate pre-profiling via LLM (speeds up execution, relying on structured CSV fields only).

The ranked results are saved to:
* `data/output/ranked_candidates.csv`
* `data/output/ranked_candidates.json`

### 2. Streamlit Web App (Recommended)
To launch the premium visual dashboard:
```bash
streamlit run app/streamlit_app.py
```
This will open the application in your browser. You can paste any JD, upload your CSV file, adjust weights dynamically, and download the resulting CSV.

---

## 🧪 Running Tests
To verify implementation logic:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
