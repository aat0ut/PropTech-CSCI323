import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import TEST_DATA, fmt_aed
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Market Trends", page_icon="📊", layout="wide")

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

h1 { color: #1E293B !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { color: #1E293B !important; font-weight: 600 !important; }

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
</style>
""", unsafe_allow_html=True)

st.title("Dubai Property Market Trends")
st.markdown("Insights derived from the model's test predictions and training data.")
st.markdown("---")

df = TEST_DATA.copy()

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Dataset Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Records",      f"{len(df):,}")
m2.metric("Median Price",       fmt_aed(df["predicted"].median()))
m3.metric("Avg Price",          fmt_aed(df["predicted"].mean()))
m4.metric("Price Std Dev",      fmt_aed(df["predicted"].std()))

st.markdown("---")

# ── Chart 1: Price distribution ───────────────────────────────────────────────
st.subheader("Price Distribution")
st.caption("How predicted property prices are distributed across the dataset.")

fig1 = px.histogram(
    df,
    x="predicted",
    nbins=60,
    color_discrete_sequence=["#2563EB"],
    labels={"predicted": "Predicted Price (AED)"},
    title="Distribution of Predicted Property Prices",
)
fig1.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(tickformat=",.0f"),
    bargap=0.05,
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Chart 2: Actual vs Predicted ──────────────────────────────────────────────
st.subheader("Actual vs Predicted Prices")
st.caption("Each dot is one property from the test set. Perfect predictions sit on the diagonal line.")

sample = df.sample(min(500, len(df)), random_state=42)
max_val = max(df["actual"].max(), df["predicted"].max())

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=sample["actual"],
    y=sample["predicted"],
    mode="markers",
    marker=dict(color="#2563EB", size=5, opacity=0.6),
    name="Predictions",
))
fig2.add_trace(go.Scatter(
    x=[0, max_val],
    y=[0, max_val],
    mode="lines",
    line=dict(color="#EF4444", dash="dash", width=2),
    name="Perfect Prediction Line",
))
fig2.update_layout(
    title="Actual vs Predicted Property Prices",
    xaxis_title="Actual Price (AED)",
    yaxis_title="Predicted Price (AED)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(tickformat=",.0f"),
    yaxis=dict(tickformat=",.0f"),
    height=500,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Chart 3: Confidence interval widths ───────────────────────────────────────
st.subheader("Confidence Interval Widths")
st.caption("How wide are the model's confidence intervals? Narrower = more confident.")

df["ci_width"] = df["ci_high"] - df["ci_low"]

fig3 = px.histogram(
    df,
    x="ci_width",
    nbins=50,
    color_discrete_sequence=["#22C55E"],
    labels={"ci_width": "Confidence Interval Width (AED)"},
    title="Distribution of Confidence Interval Widths",
)
fig3.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(tickformat=",.0f"),
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Chart 4: Base model comparison ────────────────────────────────────────────
st.subheader("Base Model Comparison")
st.caption("Comparing predictions from each individual model in the ensemble.")

if all(c in df.columns for c in ["xgb_pred", "lgbm_pred", "rf_pred"]):
    sample2 = df.sample(min(300, len(df)), random_state=1)
    fig4 = go.Figure()
    for col, color, name in [
        ("xgb_pred",  "#2563EB", "XGBoost"),
        ("lgbm_pred", "#22C55E", "LightGBM"),
        ("rf_pred",   "#F59E0B", "Random Forest"),
    ]:
        fig4.add_trace(go.Scatter(
            x=sample2["actual"],
            y=sample2[col],
            mode="markers",
            marker=dict(size=4, opacity=0.5, color=color),
            name=name,
        ))
    fig4.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#94A3B8", dash="dash", width=1),
        name="Perfect Line",
    ))
    fig4.update_layout(
        title="Individual Model Predictions vs Actual",
        xaxis_title="Actual Price (AED)",
        yaxis_title="Predicted Price (AED)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(tickformat=",.0f"),
        yaxis=dict(tickformat=",.0f"),
        height=500,
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Base model columns not available.")

st.markdown("---")

# ── Accuracy table ────────────────────────────────────────────────────────────
st.subheader("Model Accuracy by Property Type")
st.caption("MAPE = Mean Absolute Percentage Error. Lower is better.")

acc_data = {
    "Property Type": ["Units", "Villas", "Land", "Buildings"],
    "MAPE":          ["12%",   "13%",    "24%",  "~15%"],
    "Confidence":    ["High",  "High",   "Medium", "Medium"],
    "Note":          [
        "Most reliable — largest dataset",
        "Good accuracy — diverse properties",
        "Higher variance — fewer transactions",
        "Moderate — mixed use cases",
    ]
}
st.dataframe(pd.DataFrame(acc_data), use_container_width=True, hide_index=True)