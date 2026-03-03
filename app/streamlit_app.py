import sys
sys.path.append('../src')

import streamlit as st
from predict import predict

# --- Page config ---
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Job Scam Detector")
st.caption("Paste a job posting below to check if it's suspicious")
st.divider()

# --- Input form ---
description = st.text_area("Job Description *", height=200,
    placeholder="Paste the full job description here...")

col1, col2 = st.columns(2)
with col1:
    requirements = st.text_area("Requirements", height=100,
        placeholder="Skills, qualifications needed...")
with col2:
    benefits = st.text_area("Benefits", height=100,
        placeholder="What they're offering...")

salary = st.text_input("Salary Range", placeholder="e.g. $50,000 - $70,000")

col3, col4 = st.columns(2)
with col3:
    has_logo = st.checkbox("Job posting has a company logo")
with col4:
    has_profile = st.checkbox("Company profile is provided")

st.divider()
analyze = st.button("🔎 Analyze Job Posting", use_container_width=True)

# --- Prediction ---
if analyze:
    if not description.strip():
        st.error("Please paste a job description first.")
    else:
        with st.spinner("Analyzing..."):
            prob, reasons = predict(
                description=description,
                has_logo=has_logo,
                has_company_profile=has_profile,
                salary_range=salary,
                requirements=requirements,
                benefits=benefits
            )

        st.divider()

        # Score display
        pct = prob * 100
        st.metric("Scam Probability", f"{pct:.1f}%")
        st.progress(prob)

        if prob >= 0.7:
            st.error("🚨 HIGH RISK — This posting shows strong scam signals")
        elif prob >= 0.4:
            st.warning("⚠️ MODERATE RISK — Proceed with caution")
        else:
            st.success("✅ LOW RISK — This posting appears legitimate")

        # Reasons
        st.subheader("Why this score?")
        for r in reasons:
            st.write(r)