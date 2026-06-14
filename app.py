import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import chromadb
except Exception:
    chromadb = None

load_dotenv()

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def extract_keywords(text: str) -> List[str]:
    common_skills = [
        "python", "sql", "excel", "power bi", "tableau", "machine learning", "deep learning",
        "nlp", "generative ai", "gen ai", "llm", "rag", "langchain", "llamaindex",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "fastapi", "flask", "streamlit", "docker", "github", "aws", "azure", "gcp",
        "statistics", "hypothesis testing", "regression", "classification", "clustering",
        "data visualization", "eda", "feature engineering", "model deployment", "api",
        "vector database", "chromadb", "pinecone", "qdrant", "embeddings", "transformers"
    ]
    text_lower = text.lower()
    found = []
    for skill in common_skills:
        if skill in text_lower:
            found.append(skill)
    return sorted(set(found))


def compute_ats_score(resume_text: str, jd_text: str) -> Tuple[int, List[str], List[str]]:
    resume_skills = extract_keywords(resume_text)
    jd_skills = extract_keywords(jd_text)
    if not jd_skills:
        return 50, resume_skills, []
    matched = [skill for skill in jd_skills if skill in resume_skills]
    missing = [skill for skill in jd_skills if skill not in resume_skills]
    score = int(round((len(matched) / len(jd_skills)) * 100))
    return score, matched, missing


def retrieve_relevant_chunks(resume_text: str, jd_text: str, top_k: int = 5) -> List[str]:
    model = load_embedding_model()
    chunks = chunk_text(resume_text)
    if not chunks:
        return []
    chunk_embeddings = model.encode(chunks)
    jd_embedding = model.encode([jd_text])
    sims = cosine_similarity(jd_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return f"Gemini generation failed: {exc}"


def fallback_analysis(score: int, matched: List[str], missing: List[str], context_chunks: List[str]) -> str:
    matched_text = ", ".join(matched) if matched else "No strong direct keyword matches found."
    missing_text = ", ".join(missing) if missing else "No major missing skills detected from the keyword list."
    evidence = "\n\n".join(context_chunks[:3])
    return f"""
### ATS Match Summary
Your estimated ATS match score is **{score}/100**.

### Matched Skills
{matched_text}

### Missing / Weak Skills
{missing_text}

### Resume Improvement Suggestions
- Add missing JD keywords naturally in your Skills and Projects sections.
- Rewrite project bullets with measurable impact, tools used, and business outcome.
- Add a dedicated GenAI/RAG project section if targeting AI/Data Science roles.
- Include deployment details such as Streamlit, FastAPI, Docker, or cloud hosting.

### Strong Project Bullet Example
Built a RAG-based GenAI application using Python, LangChain, ChromaDB, embeddings, and Streamlit to analyze resumes against job descriptions, identify skill gaps, generate ATS scores, and provide grounded recommendations.

### Interview Questions You Should Prepare
1. How did you build the RAG pipeline?
2. Why did you choose your embedding model and vector database?
3. How did you reduce hallucinations?
4. How did you evaluate retrieval quality?
5. How would you deploy this for real users?

### Evidence From Resume Context
{evidence}
"""


def generate_ai_report(resume_text: str, jd_text: str, score: int, matched: List[str], missing: List[str], context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert resume reviewer for Data Science and GenAI roles.
Use only the provided resume context as evidence. Do not invent experience.

Resume Context:
{context}

Job Description:
{jd_text}

ATS Score: {score}/100
Matched skills: {matched}
Missing skills: {missing}

Generate a structured report with:
1. ATS match summary
2. Matched strengths
3. Missing skills
4. Resume improvement suggestions
5. 5 improved resume bullets in STAR/impact format
6. 7 interview questions based on the JD
7. Evidence-based answer strategy using only resume context
"""
    generated = call_gemini(prompt)
    if generated and not generated.startswith("Gemini generation failed"):
        return generated
    return fallback_analysis(score, matched, missing, context_chunks)


def save_report_pdf(report_text: str) -> Path:
    pdf_path = OUTPUT_DIR / "AI_Career_Copilot_Report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for line in report_text.split("\n"):
        if line.strip().startswith("###"):
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>{line.replace('###', '').strip()}</b>", styles["Heading2"]))
        elif line.strip():
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_line, styles["BodyText"]))
            story.append(Spacer(1, 6))
    doc.build(story)
    return pdf_path


st.title("🚀 AI Career Copilot")
st.caption("RAG-based resume analyzer for ATS score, skill gap analysis, grounded suggestions, and interview preparation.")

with st.sidebar:
    st.header("Project Controls")
    top_k = st.slider("Number of resume chunks to retrieve", 3, 8, 5)
    st.info("Add GOOGLE_API_KEY in .env for Gemini generation. Without it, the app uses a strong rule-based fallback.")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col2:
    jd_text = st.text_area("Paste Job Description", height=260, placeholder="Paste the job description here...")

analyze = st.button("Analyze Resume", type="primary")

if analyze:
    if not resume_file:
        st.error("Please upload a resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing resume with RAG pipeline..."):
            resume_text = clean_text(extract_pdf_text(resume_file))
            jd_clean = clean_text(jd_text)
            score, matched, missing = compute_ats_score(resume_text, jd_clean)
            context_chunks = retrieve_relevant_chunks(resume_text, jd_clean, top_k=top_k)
            report = generate_ai_report(resume_text, jd_clean, score, matched, missing, context_chunks)
            pdf_path = save_report_pdf(report)

        st.subheader("ATS Match Score")
        st.progress(score / 100)
        st.metric("Estimated Score", f"{score}/100")

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.subheader("Matched Skills")
            st.write(", ".join(matched) if matched else "No direct skill matches found.")
        with score_col2:
            st.subheader("Missing Skills")
            st.write(", ".join(missing) if missing else "No major missing skills detected.")

        st.subheader("AI Career Report")
        st.markdown(report)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f,
                file_name="AI_Career_Copilot_Report.pdf",
                mime="application/pdf"
            )

st.divider()
st.markdown("""
### Resume-Worthy Features Implemented
- PDF resume parsing
- Resume chunking
- Semantic retrieval using embeddings
- JD-to-resume similarity matching
- ATS keyword score
- Missing skill detection
- Grounded recommendation generation
- PDF report export
""")
