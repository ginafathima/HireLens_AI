# HireLens AI – AI-Powered Resume Evaluation System

**Assessment Track:** Applied AI & NLP Engineering (`ASSIGN-AI-01`)  
**Target Role:** Python AI & Data Science Trainee / Intern  
**Pylint Code Quality Score:** **10.00 / 10**  
**Test Suite:** **17 passed** (100% test pass rate)  

---

## 1. Project Overview

**HireLens AI** is a production-grade, token-efficient Python document intelligence pipeline designed to evaluate candidate resumes against benchmark Job Descriptions (JDs). It parses multi-page PDF and DOCX files, performs noise reduction and token budgeting, calculates lexical/semantic overlap using `scikit-learn` TF-IDF cosine similarity, queries free-tier LLMs via OpenRouter with exponential backoff and model fallbacks, and programmatically computes deterministic weighted match scores validated by Pydantic.

### Why HireLens AI?
* **Zero Hallucinated Arithmetic:** Many AI evaluation tools ask the LLM to compute the weighted score directly, leading to arithmetic hallucinations and inconsistent candidate ranking. HireLens AI tasks the LLM only with dimensional rating (0–100) and qualitative evidence, then **programmatically computes the overall score using Python math**.
* **Strict Pylint Compliance:** Engineered to the highest software craftsmanship standards, achieving a verified **10.00 / 10** Pylint score across all source files and test suites.
* **Token Economics & Bonus Pre-Matching:** Incorporates local TF-IDF vectorization and cosine similarity to estimate candidate fit and identify overlapping technical terminology prior to LLM invocation, saving API tokens and providing recruiter transparency.

---

## 2. Key Features

1. **Multi-Format Document Parsing:**
   * **PDF Ingestion:** Extracts multi-page text using `pypdf`, handling encrypted documents and filtering empty pages.
   * **DOCX Ingestion:** Extracts paragraphs and table structures (where skills and coursework are often located) using `python-docx`.
   * **Defensive Error Handling:** Provides beginner-friendly exceptions for missing files, empty documents, corrupted files, and unsupported formats.

2. **Text Preprocessing & Token Budgeting:**
   * Normalizes Unicode characters and non-breaking spaces.
   * Removes repetitive page numbers (`Page 1 of 3`), decorative line dividers, and unusual whitespace.
   * Standardizes bullet point glyphs into consistent markdown dashes.
   * Implements token estimation (~4 characters/token) and safe paragraph-aware truncation to prevent context window overflow.

3. **Advanced Semantic Pre-Matching (15-Point Bonus Rubric):**
   * Uses `scikit-learn`'s `TfidfVectorizer` (unigrams + bigrams, English stop words removed) and `cosine_similarity`.
   * Computes preliminary lexical overlap percentage (e.g., strong candidate ~22.4% vs. weak candidate ~2.5%).
   * Automatically extracts top shared technical keywords between the JD and the resume.

4. **Resilient OpenRouter AI Integration:**
   * Native HTTP client using Python `requests.Session`.
   * Never hardcodes credentials; loads `OPENROUTER_API_KEY` from `.env`.
   * Automatic exponential backoff retry on HTTP 429 (rate limits) and 5xx server errors.
   * **Model Fallback Cascade:** Automatically transitions to alternative free-tier models if the primary model is busy or rate-limited.
   * Enforces JSON Schema mode and strips markdown code fences (` ```json `).

5. **Deterministic Weighted Scoring:**
   * Evaluates across the 5 mandatory technical dimensions.
   * Applies the exact mathematical formula in Python.
   * Classifies candidates into **Strong Match** ($\ge 80$), **Moderate Match** ($60 - 79$), or **Weak Match** ($< 60$).

6. **Pydantic Validation & Dual Reporting:**
   * Enforces score bounds (0–100) and required field types.
   * Generates `evaluation_report.json` conforming to Section 3.4 of the assessment.
   * Prints a formatted ASCII console dashboard with color badges and key takeaways.

---

## 3. Mandatory Evaluation Dimensions & Formula

| Dimension | JSON Key | Weight | Focus Areas |
| :--- | :--- | :---: | :--- |
| **Core Skills Match** | `core_skills` | **30%** | Python 3.10+, Pandas, NumPy, pypdf, docx, requests, REST APIs, prompt engineering |
| **Projects & Practical Implementation** | `projects_and_assignments` | **25%** | Project depth, complexity, deployed systems vs. generic tutorials |
| **Experience & Internships** | `experience_internships` | **15%** | Real-world problem solving, engineering internships, open-source work |
| **Academic Background & Foundations** | `academic_qualification` | **15%** | CS/AI/Math degree, 2024–2026 graduation batch eligibility |
| **Resume Structure & Quality** | `resume_quality` | **15%** | Formatting clarity, quantifiable impact metrics, concise language |
| **Total** | | **100%** | |

### Programmatic Scoring Formula:
$$\text{Overall Score} = \operatorname{round}\left( 0.30 \cdot S_{\text{skills}} + 0.25 \cdot S_{\text{projects}} + 0.15 \cdot S_{\text{experience}} + 0.15 \cdot S_{\text{academics}} + 0.15 \cdot S_{\text{quality}} \right)$$

---

## 4. Repository Structure

```
HireLens_AI/
├── .env.example               # Template for environment variables and model settings
├── .env                       # Local secrets (ignored by Git)
├── .gitignore                 # Excludes .env, .venv, caches, and generated reports
├── README.md                  # Comprehensive documentation and evaluation report
├── requirements.txt           # Pinned production dependencies
├── jd.txt                     # Official benchmark job description
├── main.py                    # CLI entrypoint and orchestrator
├── generate_sample_resumes.py # Script used to generate synthetic test resumes
├── sample_resumes/
│   ├── candidate_strong.pdf   # Strong fit candidate (Python AI / Data Science Trainee)
│   └── candidate_weak.docx    # Weak fit candidate (B.Com Accounting / Data Entry)
├── src/
│   ├── __init__.py            # Package metadata
│   ├── config.py              # Environment settings, weights, and model cascades
│   ├── parsers.py             # Modular PDF and DOCX text extraction utilities
│   ├── preprocessor.py        # Text cleaning, token budgeting, and TF-IDF pre-matching
│   ├── openrouter_client.py   # Resilient OpenRouter client with retries and fallbacks
│   └── evaluator.py           # Pydantic schemas, deterministic math, and report writer
└── tests/
    ├── __init__.py
    ├── test_parsers.py        # PDF/DOCX parsing and error handling tests
    ├── test_preprocessor.py   # Normalization, token bounds, and TF-IDF tests
    ├── test_evaluator.py      # Deterministic math, Pydantic validation, and report tests
    └── test_openrouter_client.py # Markdown fence cleaning, auth, and retry tests
```

---

## 5. Installation & Setup Guide

### Prerequisites
* Python 3.10 or higher (Tested on Python 3.13)
* Standard Python `venv`

### Step 1: Clone or Navigate to the Project
```bash
cd C:\Users\USER\HireLens_AI
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Environment Configuration

1. Copy the example configuration to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your OpenRouter API key:
   ```env
   # Get a free key from https://openrouter.ai/keys
   OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key

   # Primary Model (Recommended: Google Gemini 2.0 Flash Lite Free)
   OPENROUTER_MODEL=google/gemini-2.0-flash-lite-001

   # Fallback models attempted automatically on 429 rate limit or outage
   OPENROUTER_FALLBACK_MODELS=meta-llama/llama-3.3-70b-instruct:free,google/gemma-2-9b-it:free,mistralai/mistral-7b-instruct:free

   # Request Timeout in seconds
   OPENROUTER_TIMEOUT=45

   # Retry attempts per model
   OPENROUTER_MAX_RETRIES=3
   ```

> **Security Note:** `.env` is listed in `.gitignore` and is never committed to source control.

---

## 7. Execution Guide

### Evaluating the Strong Fit Candidate (PDF)
```bash
python main.py --resume sample_resumes/candidate_strong.pdf --jd jd.txt --output evaluation_report.json
```

### Evaluating the Weak Fit Candidate (DOCX)
```bash
python main.py --resume sample_resumes/candidate_weak.docx --jd jd.txt --output evaluation_report.json
```

### Optional CLI Arguments
* `--model <model_id>`: Override the default AI model on the fly.
* `--output <path>`: Customize the output JSON report destination.
* `--verbose` (`-v`): Enable detailed debug logs (HTTP payloads, backoff sleep times).

---

## 8. OpenRouter Free Model Selection & Tradeoffs

HireLens AI defaults to `google/gemini-2.0-flash-lite-001` (or `meta-llama/llama-3.3-70b-instruct:free`) on OpenRouter for several strategic reasons:

1. **Structured JSON Adherence:** Strong adherence to `response_format: {"type": "json_object"}`, avoiding schema drift.
2. **Context Efficiency:** Lightweight token consumption while maintaining deep technical comprehension of Python libraries and graduation criteria.
3. **Low Latency & High Rate Limits:** Offers quick evaluation cycles without hitting aggressive throttling.
4. **Fallback Resilience:** If the primary endpoint experiences transient server errors (HTTP 500/502/503) or rate limits (HTTP 429), our client automatically rolls over to `meta-llama/llama-3.3-70b-instruct:free`, `google/gemma-2-9b-it:free`, or `mistralai/mistral-7b-instruct:free`.

---

## 9. Example Output

### Console Dashboard
```
============================================================================
  HIRELENS AI - CANDIDATE EVALUATION DASHBOARD
============================================================================
  Candidate Name:       Aarav Sharma
  Overall Match Score:  87/100
  Fit Status:           Strong Match
  TF-IDF Pre-Score:     22.4% lexical overlap
----------------------------------------------------------------------------
  CATEGORY                       WEIGHT     SCORE      STATUS
----------------------------------------------------------------------------
  Core Skills                    30%        90         [EXCELLENT]
  Projects And Assignments       25%        85         [EXCELLENT]
  Experience Internships         15%        80         [EXCELLENT]
  Academic Qualification         15%        95         [EXCELLENT]
  Resume Quality                 15%        85         [EXCELLENT]
----------------------------------------------------------------------------
  KEY STRENGTHS:
   + Demonstrated hands-on experience with Python 3.11, pypdf, and python-docx.
   + Built and deployed DocuIntellect end-to-end with FastAPI and RAG architecture.
   + Strong academic CS pedigree (KTU 2025 batch, CGPA 8.85/10.0).

  SKILL GAPS & AREAS FOR IMPROVEMENT:
   - Limited exposure to local inference runtimes (e.g., Ollama).
   - No direct experience listed with Django/HTMX.

  HIRING MANAGER RECOMMENDATION:
   >> Advance to technical interview round with a focus on embeddings and RAG design.
============================================================================
[SUCCESS] Report written to: evaluation_report.json
```

### Generated `evaluation_report.json`
```json
{
  "candidate_name": "Aarav Sharma",
  "overall_match_score": 87,
  "fit_status": "Strong Match",
  "category_scores": {
    "core_skills": {
      "score": 90,
      "weight": "30%",
      "justification": "Exceptional match with Python, pypdf, python-docx, FastAPI, and scikit-learn."
    },
    "projects_and_assignments": {
      "score": 85,
      "weight": "25%",
      "justification": "DocuIntellect project shows applied ML, vector indexing with ChromaDB, and high test coverage."
    },
    "experience_internships": {
      "score": 80,
      "weight": "15%",
      "justification": "Completed a 6-month internship in applied AI engineering focusing on document parsing."
    },
    "academic_qualification": {
      "score": 95,
      "weight": "15%",
      "justification": "B.Tech in Computer Science (2025 batch) with high CGPA (8.85/10) and relevant coursework."
    },
    "resume_quality": {
      "score": 85,
      "weight": "15%",
      "justification": "Clean bullet points with quantifiable impact metrics (34% token reduction, 15k documents)."
    }
  },
  "key_strengths": [
    "Demonstrated hands-on experience with Python 3.11, pypdf, and python-docx.",
    "Built and deployed DocuIntellect end-to-end with FastAPI and RAG architecture.",
    "Strong academic CS pedigree (KTU 2025 batch, CGPA 8.85/10.0)."
  ],
  "skill_gaps": [
    "Limited exposure to local inference runtimes (e.g., Ollama).",
    "No direct experience listed with Django/HTMX."
  ],
  "hiring_manager_recommendation": "Advance to technical interview round with a focus on embeddings and RAG design."
}
```

---

## 10. Automated Testing & Quality Verification

### Running the Test Suite
The project contains 17 comprehensive automated unit and integration tests covering all critical components:

```bash
pytest tests/ -v
```

### Test Coverage Highlights:
* `test_parsers.py`: Validates extraction accuracy from both PDF and DOCX, and confirms appropriate exception handling for missing or unsupported formats.
* `test_preprocessor.py`: Verifies Unicode normalization, whitespace collapsing, token estimation heuristics, paragraph-aware truncation, and TF-IDF discrimination.
* `test_evaluator.py`: Tests the mathematical calculation of deterministic scores, Pydantic field validation, score clamping, and JSON report persistence.
* `test_openrouter_client.py`: Verifies markdown code fence cleaning, missing API key detection, HTTP 429 exponential backoff retries, and fallback model triggering.

### Pylint Verification
Run Pylint across the entire codebase:
```bash
pylint src/ main.py tests/
```

**Official Pylint Output:**
```
-------------------------------------------------------------------
Your code has been rated at 10.00/10
```

---

## 11. AI Tooling & Development Disclosure

In accordance with Section 4 and Section 8 of the assessment brief:
* **Development Assistant:** Antigravity (Powered by Google Gemini 3.8 Flash High) was used as an AI pair programmer and architecture consultant during development.
* **Engineering Principles Applied:** All prompts, data structures, error recovery cascades, and mathematical scoring logic were engineered modularly and manually reviewed for strict adherence to PEP 8, deterministic accuracy, and token economics.
