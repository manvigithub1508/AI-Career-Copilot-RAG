# AI Career Copilot

A RAG-based GenAI resume analyzer that compares a resume with a job description and generates:

- ATS match score
- Missing skills
- Resume improvement suggestions
- STAR-format bullet improvements
- Interview questions
- Evidence-based answers grounded in resume content

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Gemini optional

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

Optional Gemini setup:

```bash
copy .env.example .env
# Add GOOGLE_API_KEY in .env
```

Run:

```bash
streamlit run app.py
```

## How It Works

1. Upload a resume PDF.
2. Paste a job description.
3. The app extracts resume text.
4. Resume content is chunked and embedded.
5. Chroma retrieves the most relevant resume sections.
6. The LLM or fallback rule-based engine generates grounded insights.

## Resume Project Bullet

Built a RAG-based GenAI resume optimization system using LangChain, ChromaDB, sentence-transformer embeddings, and Streamlit to analyze resumes against job descriptions, identify skill gaps, generate ATS scores, and produce evidence-grounded career recommendations.
