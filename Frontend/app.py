import streamlit as st

st.set_page_config(
    page_title="Dubai AVM — Property Valuation",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
mainmenu
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
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
    font-size: 26px !important;
    font-weight: 700 !important;
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

.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #1D4ED8 !important;
}

.badge {
    display: inline-block;
    background-color: #EFF6FF;
    color: #2563EB;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<span class="badge">CSCI 323 · UOWD · 2025</span>', unsafe_allow_html=True)
st.title("Dubai Property Valuation")
st.markdown("AI-powered automated valuation model built on Dubai Land Department data.")
st.divider()

# ── Two columns ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("What is this tool?")
    st.markdown("""
    This tool uses a trained machine learning model to estimate property values
    across Dubai — instantly, transparently, and honestly.

    Built on **Dubai Land Department** transaction data, our ensemble model
    combines XGBoost, LightGBM, and Random Forest to deliver reliable valuations
    with confidence intervals — not just a single number.
    """)

    st.divider()

    st.subheader("How to use this app")
    st.markdown("""
    **1. Valuation** — Enter property details and get an estimated price with a confidence range.

    **2. Explanation** — See which factors drove the price up or down, in plain English.

    **3. Market Trends** — Explore aggregate insights from the Dubai property market.

    **4. About** — Learn about the model, its accuracy, and its limitations.
    """)

with col2:
    st.subheader("Model Snapshot")
    st.metric("Overall Accuracy (MAPE)", "~13%")
    st.metric("Property Types Covered", "4")
    st.metric("Data Source", "DLD")
    st.metric("Model Type", "Ensemble")

st.divider()

# ── Accuracy ──────────────────────────────────────────────────────────────────
st.subheader("Model Accuracy by Property Type")
st.caption("MAPE = Mean Absolute Percentage Error — lower is better.")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Units",     "12% MAPE")
a2.metric("Villas",    "13% MAPE")
a3.metric("Land",      "24% MAPE")
a4.metric("Buildings", "~15% MAPE")

st.divider()

st.warning("This tool is for informational purposes only. Predictions are estimates based on historical data and should not be used as the sole basis for any financial or legal decision. Always consult a licensed real estate professional.")

st.caption("PropTech AVM · CSCI 323 · University of Wollongong Dubai · 2025")
