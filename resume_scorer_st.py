import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import pdfplumber
import os
from dotenv import load_dotenv

# Load Groq API Key
load_dotenv()
groq_apikey = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume Scorer", page_icon="📊", layout="centered")
st.title("📊 Resume Scorer vs Job Description")
st.write("Evaluate how well your resume matches a specific Job Description")

if not groq_apikey:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.info("Please add `GROQ_API_KEY=your_key_here` in your `.env` file")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

if uploaded_file:
    # Extract resume text
    with st.spinner("Extracting text from resume..."):
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            resume_context = "\n\n".join(pages)

    if not resume_context.strip():
        st.error("Could not extract text from the PDF. Please use a searchable PDF.")
        st.stop()

    st.success("✅ Resume loaded successfully!")

    # Job Description Input
    st.subheader("📋 Job Description")
    job_title = st.text_input("Job Title", placeholder="e.g. AI Engineer / Data Scientist")
    company_name = st.text_input("Company Name (Optional)", placeholder="e.g. Google")
    
    job_description = st.text_area(
        "Paste the Full Job Description *",
        height=250,
        placeholder="Paste the complete job description here..."
    )

    if st.button("🔍 Score Resume Against JD", type="primary"):
        if not job_description.strip():
            st.warning("Please paste the Job Description")
        else:
            with st.spinner("Analyzing resume against Job Description..."):
                try:
                    llm = ChatGroq(
                        model="groq/compound-mini",
                        api_key=groq_apikey,
                        temperature=0.3
                    )

                    scorer_prompt = PromptTemplate(
                        input_variables=["resume", "job_title", "company_name", "job_description"],
                        template="""
You are an expert ATS and technical recruiter. Score the candidate's resume **strictly with respect to the given Job Description**.

Job Title: {job_title}
Company: {company_name}
Job Description: {job_description}

Resume:
{resume}

Evaluate with **EXACT** structure:

1. **Overall Match Score**: X/100
2. **Key Strengths**: 
   • ...(list at least 3 strong matches)
3. **Critical Gaps / Weaknesses**: 
   • ...(list at least 3 missing or weak areas)
4. **Skills Match**:
   • **Required Skills Present**: ...
   • **Missing Skills**: ...
5. **Experience Relevance**: ...
6. **Recommendations to Improve Match**: 
   • ...(actionable suggestions)

Be honest, critical, and constructive.
"""
                    )

                    formatted_prompt = scorer_prompt.format(
                        resume=resume_context,
                        job_title=job_title,
                        company_name=company_name or "Not Provided",
                        job_description=job_description
                    )

                    # Streaming Output
                    st.subheader("📋 Resume vs JD Evaluation")
                    response_placeholder = st.empty()
                    full_response = ""

                    for chunk in llm.stream(formatted_prompt):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("👆 Please upload your resume PDF to continue")

st.caption("Powered by Groq • Simple & Accurate Resume Scoring")