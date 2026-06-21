import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import explain_property, fmt_aed, fmt_aed_compact
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Explanation", page_icon="🔍", layout="wide")

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

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

.summary-box {
    background-color: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
}
.summary-box p {
    color: #1E293B;
    margin: 6px 0;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Why This Price?")
st.markdown("A breakdown of every factor that pushed the price up or down.")
st.divider()

# ── Check prediction exists ───────────────────────────────────────────────────
if "last_prediction" not in st.session_state or "last_features" not in st.session_state:
    st.info("You haven't run a valuation yet. Head to the **Valuation** page first, then come back here.")
    st.stop()

prediction = st.session_state["last_prediction"]
features   = st.session_state["last_features"]
pred_value = prediction["predicted_value_aed"]

st.success(f"Explaining prediction of **{fmt_aed(pred_value)}** for a **{features['prop_type']}** in **{features['area']}**")
st.divider()

# ── Get SHAP values ───────────────────────────────────────────────────────────
with st.spinner("Calculating feature contributions..."):
    try:
        explanation = explain_property(features, pred_value)
        shap_values = explanation["shap_values"]
        base_value  = explanation["base_value"]
    except Exception as e:
        st.error(f"Could not load explanation: {e}")
        st.stop()

# ── Feature labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "area":                          "Location (Area)",
    "procedure_area":                "Property Size",
    "prop_type_Unit":                "Property Type: Unit",
    "prop_type_Villa":               "Property Type: Villa",
    "prop_type_Land":                "Property Type: Land",
    "prop_sb_type":                  "Property Sub-Type",
    "rooms":                         "Number of Rooms",
    "nearest_metro":                 "Nearest Metro",
    "nearest_mall":                  "Nearest Mall",
    "nearest_landmark":              "Nearest Landmark",
    "is_offplan":                    "Off-Plan Status",
    "has_parking":                   "Parking",
    "usage_Residential":             "Usage: Residential",
    "usage_Commercial":              "Usage: Commercial",
    "usage_Hospitality_":            "Usage: Hospitality",
    "usage_Industrial":              "Usage: Industrial",
    "usage_Multi_Use":               "Usage: Multi-Use",
    "usage_Other":                   "Usage: Other",
    "usage_Storage":                 "Usage: Storage",
    "master_project":                "Master Project",
    "project":                       "Project",
    "community_30d_avg_price":       "Community Avg Price (30d)",
    "off_plan_vol_ratio":            "Off-Plan Volume Ratio",
    "date_year":                     "Year",
    "month_sin":                     "Month (Seasonal)",
    "month_cos":                     "Month (Seasonal)",
    "procedure_Sell":                "Procedure: Sell",
    "procedure_Sell___Pre_registration": "Procedure: Pre-Registration",
    "procedure_Sell_Development":    "Procedure: Development",
    "procedure_Other":               "Procedure: Other",
    "other_factors":                 "Other Factors",
}

# ── Sort & top 7 ─────────────────────────────────────────────────────────────
sorted_shap  = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
top_features = sorted_shap[:7]
other_sum    = sum(v for _, v in sorted_shap[7:])
if other_sum != 0:
    top_features.append(("other_factors", other_sum))

labels   = [FEATURE_LABELS.get(k, k) for k, _ in top_features]
values   = [round(v, 2) for _, v in top_features]
measures = ["absolute"] + ["relative"] * len(values) + ["total"]
x_plot   = [round(base_value)] + values + [round(pred_value)]
y_plot   = ["City Base Price"] + labels + ["Final Price"]

# ── Waterfall chart ───────────────────────────────────────────────────────────
fig = go.Figure(go.Waterfall(
    orientation="h",
    measure=measures,
    x=x_plot,
    y=y_plot,
    connector={"line": {"color": "#E2E8F0", "width": 1}},
    increasing={"marker": {"color": "#22C55E"}},
    decreasing={"marker": {"color": "#EF4444"}},
    totals={"marker":    {"color": "#2563EB"}},
    texttemplate="%{x:+,.0f}",
    textposition="outside",
))

fig.update_layout(
    title="Feature Contributions to Final Price (AED)",
    xaxis_title="AED Contribution",
    height=500,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1E293B", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(tickformat=",.0f", gridcolor="#F1F5F9"),
    title_font=dict(size=16, color="#1E293B", weight=700),
)

st.plotly_chart(fig, use_container_width=True)

# ── Legend ────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.markdown("🟢 **Green** = pushed price UP")
c2.markdown("🔴 **Red** = pushed price DOWN")
c3.markdown("🔵 **Blue** = final estimated price")

st.divider()

# ── Plain English ─────────────────────────────────────────────────────────────
st.subheader("In Plain English")

top3 = sorted_shap[:3]
summary_lines = []
for feat, val in top3:
    label     = FEATURE_LABELS.get(feat, feat)
    direction = "added" if val >= 0 else "reduced"
    summary_lines.append(f"<p><b>{label}</b> {direction} <b>{fmt_aed_compact(abs(val))}</b> to the estimated value.</p>")

st.markdown(f"""
<div class="summary-box">
    <p>The base city-average price for this property type starts at <b>{fmt_aed(base_value)}</b>.</p>
    <p>The three biggest factors were:</p>
    {"".join(summary_lines)}
    <p>The final estimated value lands at <b>{fmt_aed(pred_value)}</b>.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Raw table ─────────────────────────────────────────────────────────────────
with st.expander("See all feature contributions"):
    table_data = []
    for feat, val in sorted_shap:
        label     = FEATURE_LABELS.get(feat, feat)
        direction = "Adds" if val >= 0 else "Reduces"
        arrow = "⬆️" if val >= 0 else "⬇️"
        table_data.append({
            "Feature":      label,
            "Direction":    f"{arrow} {direction}",
            "Contribution": fmt_aed_compact(abs(val)),
        })
    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )