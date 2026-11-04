"""Script to generate realistic synthetic sample resumes for testing HireLens AI.

Generates:
1. sample_resumes/candidate_strong.pdf - Strong fit for Python AI & Data Science Trainee
2. sample_resumes/candidate_weak.docx - Weak fit (non-tech background, missing core skills)
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_resumes"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


class StrongResumePDF(FPDF):
    """Custom PDF generator for Strong Fit candidate."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(24, 43, 73)
        self.cell(0, 8, "AARAV SHARMA", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(90, 90, 90)
        self.cell(
            0,
            5,
            "Kochi, Kerala | aarav.sharma.dev@gmail.com | +91 98765 43210 | github.com/aarav-ai",
            ln=True,
            align="C",
        )
        self.ln(4)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 110)
        self.cell(0, 6, title.upper(), ln=True)
        self.set_draw_color(180, 190, 205)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(3)

    def bullet_point(self, bold_prefix: str, text: str):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(40, 40, 40)
        self.cell(4, 5, chr(149), ln=False)  # bullet
        self.cell(self.get_string_width(bold_prefix) + 1, 5, bold_prefix, ln=False)
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)


def generate_strong_resume_pdf():
    """Build candidate_strong.pdf."""
    pdf = StrongResumePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Professional Summary
    pdf.section_title("Professional Summary")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0,
        5,
        "Proactive Python AI & Data Science enthusiast with hands-on experience in building LLM-powered "
        "document intelligence engines, RESTful APIs, and retrieval-augmented generation (RAG) pipelines. "
        "Proficient in Python 3.11, pypdf, python-docx, scikit-learn, and prompt engineering. Strong foundation "
        "in linear algebra, algorithms, and modular OOP architecture.",
    )
    pdf.ln(3)

    # 2. Technical Skills
    pdf.section_title("Technical Skills")
    pdf.bullet_point("Languages & Core: ", "Python 3.10+, SQL, Bash, C++ (Foundations)")
    pdf.bullet_point(
        "Libraries & Data: ",
        "Pandas, NumPy, Scikit-Learn, Regular Expressions (re), pypdf, python-docx, Matplotlib",
    )
    pdf.bullet_point(
        "GenAI & LLM Tooling: ",
        "OpenRouter API, OpenAI API, HuggingFace, ChromaDB, FAISS, Prompt Engineering, Pydantic Schema Validation",
    )
    pdf.bullet_point(
        "Web & Developer Tools: ",
        "FastAPI, Requests, Git/GitHub, Docker, Linux/Unix CLI, Virtual Environments (venv), Pylint",
    )
    pdf.ln(3)

    # 3. Work & Internship Experience
    pdf.section_title("Experience & Internships")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(130, 5, "Applied AI Engineering Intern - CogniTech Labs, Kochi", ln=False)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.cell(60, 5, "Jan 2024 - Jun 2024 (6 Months)", ln=True, align="R")

    pdf.bullet_point(
        "Document Parsing Engine: ",
        "Designed modular Python ingestion pipeline extracting tabular and multi-column text from 15,000+ PDFs/DOCX using pypdf and python-docx, achieving 98.4% extraction accuracy.",
    )
    pdf.bullet_point(
        "LLM Pipeline Optimization: ",
        "Engineered token-efficient prompt templates for OpenRouter and OpenAI models with exponential backoff retries, slashing per-request token consumption by 34%.",
    )
    pdf.bullet_point(
        "FastAPI Microservice: ",
        "Integrated the evaluation workflow into high-throughput FastAPI endpoints validated strictly via Pydantic schemas, reducing API response latency to under 120ms.",
    )
    pdf.ln(3)

    # 4. Projects & Practical Implementation
    pdf.section_title("Projects & Practical Implementation")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(130, 5, "DocuIntellect - Multi-Format Document Intelligence Pipeline", ln=False)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.cell(60, 5, "Python, pypdf, ChromaDB, FastAPI", ln=True, align="R")
    pdf.bullet_point(
        "End-to-End System: ",
        "Engineered complete CLI & API pipeline to ingest, clean, chunk, and embed contracts and technical specifications. Implemented cosine similarity semantic filtering with scikit-learn.",
    )
    pdf.bullet_point(
        "Robustness & Code Hygiene: ",
        "Wrote 40+ unit tests with pytest (88% code coverage) and maintained 9.8/10 Pylint score across the codebase adhering to strict PEP 8 standards.",
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(130, 5, "Semantic Code Search Engine (RAG)", ln=False)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.cell(60, 5, "Python, OpenRouter, FAISS, Streamlit", ln=True, align="R")
    pdf.bullet_point(
        "Hybrid Search: ",
        "Combined sparse BM25 keyword matching with dense FAISS vector embeddings to query 50+ open-source Python repositories in real-time.",
    )
    pdf.ln(3)

    # 5. Academic Background
    pdf.section_title("Academic Background & Foundations")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(130, 5, "B.Tech in Computer Science & Engineering (Data Science Specialization)", ln=False)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.cell(60, 5, "2021 - 2025 (Graduating May 2025)", ln=True, align="R")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(0, 5, "Kerala Technological University (KTU) | CGPA: 8.85 / 10.0", ln=True)
    pdf.bullet_point(
        "Core Coursework: ",
        "Linear Algebra, Probability & Statistics, Data Structures & Algorithms, Natural Language Processing, Deep Learning.",
    )

    out_file = SAMPLE_DIR / "candidate_strong.pdf"
    pdf.output(str(out_file))
    print(f"[CREATED] Strong fit resume: {out_file}")


def generate_weak_resume_docx():
    """Build candidate_weak.docx."""
    doc = docx.Document()

    # Title / Header
    title = doc.add_paragraph()
    run = title.add_run("ROHAN PATEL")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0, 51, 102)

    sub = doc.add_paragraph("Ahmedabad, Gujarat | rohan.patel.accounts@yahoo.com | +91 91234 56789")
    sub.runs[0].font.size = Pt(10)
    sub.runs[0].font.color.rgb = RGBColor(100, 100, 100)

    # Objective
    doc.add_heading("Career Objective", level=2)
    doc.add_paragraph(
        "Hardworking Commerce graduate seeking an entry-level position in business operations, "
        "retail bookkeeping, or clerical data entry where I can utilize my spreadsheet management skills."
    )

    # Education
    doc.add_heading("Education", level=2)
    p_edu = doc.add_paragraph()
    r_deg = p_edu.add_run("Bachelor of Commerce (B.Com) - Financial Accounting & Taxation\n")
    r_deg.bold = True
    p_edu.add_run("Gujarat University | Graduated: 2021 | Percentage: 61%\n")
    p_edu.add_run("Relevant Subjects: Cost Accounting, Commercial Law, Auditing, Business Mathematics")

    # Skills Table
    doc.add_heading("Key Skills", level=2)
    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Category"
    hdr_cells[1].text = "Proficiencies"
    for cell in hdr_cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True

    data = [
        ("Office Tools", "Microsoft Excel (VLOOKUP, Pivot Tables), Word, PowerPoint"),
        ("Accounting Software", "Tally ERP 9, Tally Prime, Basic QuickBooks"),
        ("General Computer", "Internet browsing, Email correspondence, Typing speed 40 WPM"),
        ("Languages", "English, Hindi, Gujarati"),
    ]
    for cat, prof in data:
        row_cells = table.add_row().cells
        row_cells[0].text = cat
        row_cells[1].text = prof

    # Experience
    doc.add_heading("Work Experience", level=2)
    p_exp = doc.add_paragraph()
    r_exp_title = p_exp.add_run("Junior Accounts Assistant - Shreenath Trading Co.\n")
    r_exp_title.bold = True
    p_exp.add_run("July 2021 - Present (Ahmedabad)\n")
    doc.add_paragraph("Maintained daily purchase registers and updated inventory logbooks.", style="List Bullet")
    doc.add_paragraph("Reconciled bank statements and generated monthly tax invoices.", style="List Bullet")
    doc.add_paragraph("Filed physical receipts and organized paper documents for auditing.", style="List Bullet")

    # Projects
    doc.add_heading("Projects & Seminars", level=2)
    doc.add_paragraph(
        "Annual Budget Model: Created an automated Excel workbook to track household income and monthly expenditures using basic SUM formulas.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Attended 2-day computer literacy workshop on Windows Operating Systems and Internet safety.",
        style="List Bullet",
    )

    out_file = SAMPLE_DIR / "candidate_weak.docx"
    doc.save(str(out_file))
    print(f"[CREATED] Weak fit resume: {out_file}")


if __name__ == "__main__":
    generate_strong_resume_pdf()
    generate_weak_resume_docx()
