import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

h1 {
    color: #1E293B !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    color: #1E293B !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1E293B !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

.info-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("About This Model")
st.markdown("Everything you need to know about how this tool works and what it can and cannot do.")
st.divider()

# ── What is this ──────────────────────────────────────────────────────────────
st.subheader("What Is This Tool?")
st.markdown("""
This is an **Automated Valuation Model (AVM)** for Dubai real estate, built as part of a
PropTech project for **CSCI 323 at the University of Wollongong Dubai**.

It uses machine learning to estimate property values across Dubai based on characteristics
like location, size, property type, and proximity to key landmarks — instantly and transparently.
""")

st.divider()

# ── The Model ─────────────────────────────────────────────────────────────────
st.subheader("The Model")

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("""
    **Model Type:** Ensemble (Stacked)

    The model combines three base learners whose predictions are blended
    by a Ridge meta-learner:

    - **XGBoost** — gradient boosting, excellent on tabular data
    - **LightGBM** — fast gradient boosting, handles high-cardinality well
    - **Random Forest** — bagging ensemble, reduces variance
    - **Ridge Meta-Learner** — learns the optimal blend of the three

    Stacking means the final prediction is smarter than any single model alone.
    """)

with c2:
    st.markdown("""
    **Training Data:** Dubai Land Department (DLD)

    - Thousands of real property transactions across Dubai
    - Covers Units, Villas, Land, and Buildings
    - Includes off-plan and existing properties
    - Features: area, project, size, rooms, parking,
      usage, proximity to metro, mall, and landmark
    """)

st.divider()

# ── Accuracy ──────────────────────────────────────────────────────────────────
st.subheader("Accuracy")
st.markdown("The model is evaluated using **MAPE (Mean Absolute Percentage Error)** — the average percentage by which predictions differ from actual transaction prices.")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Units",     "12% MAPE", help="Most reliable — largest training set")
a2.metric("Villas",    "13% MAPE", help="Good accuracy — diverse properties")
a3.metric("Land",      "24% MAPE", help="Higher variance — fewer transactions")
a4.metric("Buildings", "~15% MAPE", help="Moderate — mixed use cases")

st.info("**What does 13% MAPE mean in practice?** For a property worth 2,000,000 AED, the model's prediction is typically within ±260,000 AED of the real market value. This is why we always show a confidence range — not just a single number.")

st.divider()

# ── Limitations ───────────────────────────────────────────────────────────────
st.subheader("Limitations — Read This Before Making Decisions")
st.warning("This tool is for **informational and educational purposes only**. Please understand these limitations before acting on any output.")

l1, l2 = st.columns(2, gap="large")

with l1:
    st.markdown("""
    **What the model cannot account for:**
    - Internal condition (renovated vs original)
    - Specific amenities (pool, gym, view)
    - Recent market news or economic shocks
    - Negotiation or urgency factors
    - Legal complications or disputes
    """)

with l2:
    st.markdown("""
    **When to be extra cautious:**
    - Land transactions (24% MAPE)
    - Ultra-luxury properties (limited training data)
    - Brand new projects with no comparable sales
    - Very niche or unusual locations
    """)

st.error("Do not use this tool as the sole basis for any purchase, sale, or financial decision. Always consult a licensed real estate professional and conduct proper due diligence.")

st.divider()

# ── Team ──────────────────────────────────────────────────────────────────────
st.subheader("Project Team")
st.markdown("This project was built as a group assignment for **CSCI 323 — Machine Learning** at the **University of Wollongong Dubai**.")

import pandas as pd
team_data = {
    "Role": ["Data Engineering", "Modeling", "Explainability", "Frontend / UX"],
    "Responsibility": [
        "Data collection, cleaning, and feature engineering",
        "Model training, tuning, and evaluation",
        "SHAP integration and interpretation",
        "This Streamlit application",
    ]
}
st.dataframe(pd.DataFrame(team_data), use_container_width=True, hide_index=True)

st.divider()

# ── Data source ───────────────────────────────────────────────────────────────
st.subheader("Data Source")
st.markdown("""
Transaction data sourced from the **Dubai Land Department (DLD)** —
the official government authority responsible for real estate registration in Dubai.

Data covers historical property transactions and is used strictly for
research and educational purposes.
""")

st.divider()
st.caption("PropTech AVM · CSCI 323 · University of Wollongong Dubai · 2025")