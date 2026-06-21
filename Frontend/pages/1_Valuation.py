import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from helpers import (
    predict_property, fmt_aed, fmt_aed_compact,
    area_options, project_options, master_options,
    metro_options, mall_options, landmark_options,
    PROP_TYPES, PROP_SUBTYPES, USAGES
)
import plotly.graph_objects as go

st.set_page_config(page_title="Valuation", page_icon="🏠", layout="wide")

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
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* Form styling */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Select boxes and inputs */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stCheckbox"] label {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 0 !important;
    transition: all 0.2s ease;
    width: 100%;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}

/* Result section */
.result-header {
    font-size: 14px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.result-value {
    font-size: 42px;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
}
.result-range {
    font-size: 15px;
    color: #64748B;
    margin-top: 4px;
}

/* Input borders */
[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    background-color: #F8FAFC !important;
    transition: border 0.2s ease;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #2563EB !important;
}
[data-testid="stNumberInput"] > div > div {
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    background-color: #F8FAFC !important;
}
[data-testid="stNumberInput"] > div > div:hover {
    border-color: #2563EB !important;
}
[data-testid="stCheckbox"] span {
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Property Valuation")
st.markdown("Enter the property details below to get an AI-powered price estimate.")
st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("valuation_form"):
    st.subheader("Property Details")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        area        = st.selectbox("Area",           area_options,   index=None, placeholder="Select an area...")
        master_proj = st.selectbox("Master Project", master_options, index=None, placeholder="Select master project...")
        prop_type   = st.selectbox("Property Type",  PROP_TYPES)
        usage       = st.selectbox("Usage",          USAGES)

    with col2:
        project      = st.selectbox("Project",   project_options, index=None, placeholder="Select a project...")
        prop_subtype = st.selectbox("Sub-Type",   PROP_SUBTYPES)
        size_m2      = st.number_input("Size (m²)", min_value=20,  max_value=5000, value=85, step=1)
        rooms        = st.number_input("Rooms",      min_value=0,   max_value=10,   value=2,  step=1)

    with col3:
        nearest_metro    = st.selectbox("Nearest Metro",    metro_options,    index=None, placeholder="Select metro...")
        nearest_mall     = st.selectbox("Nearest Mall",     mall_options,     index=None, placeholder="Select mall...")
        nearest_landmark = st.selectbox("Nearest Landmark", landmark_options, index=None, placeholder="Select landmark...")
        has_parking      = st.checkbox("Has Parking",       value=True)
        is_offplan       = st.checkbox("Off-Plan Property", value=False)

    submitted = st.form_submit_button("Get Valuation", use_container_width=True)

# ── Validation & Prediction ───────────────────────────────────────────────────
if submitted:
    errors = []
    if area        is None: errors.append("Please select an Area.")
    if master_proj is None: errors.append("Please select a Master Project.")
    if project     is None: errors.append("Please select a Project.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        features = {
            "area":             area,
            "project":          project,
            "master_project":   master_proj,
            "prop_type":        prop_type,
            "prop_sb_type":     prop_subtype,
            "usage":            usage,
            "procedure_area":   size_m2,
            "rooms":            rooms,
            "has_parking":      int(has_parking),
            "is_offplan":       int(is_offplan),
            "nearest_metro":    nearest_metro,
            "nearest_mall":     nearest_mall,
            "nearest_landmark": nearest_landmark,
        }

        with st.spinner("Running valuation model..."):
            try:
                result = predict_property(features)
                st.session_state["last_prediction"] = result
                st.session_state["last_features"]   = features
            except Exception as e:
                st.error(f"Something went wrong: {e}. Please try again.")
                st.stop()

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Valuation Result")

        pred   = result["predicted_value_aed"]
        ci_low = result["ci_low"]
        ci_hi  = result["ci_high"]

        # Main price display
        st.markdown(f"""
        <div class="result-header">Estimated Value</div>
        <div class="result-value">{fmt_aed(pred)}</div>
        <div class="result-range">Confidence range: {fmt_aed_compact(ci_low)} — {fmt_aed_compact(ci_hi)}</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Value", fmt_aed(pred))
        m2.metric("Low Estimate",    fmt_aed(ci_low))
        m3.metric("High Estimate",   fmt_aed(ci_hi))

        # Confidence range chart — horizontal
        # Confidence range chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Confidence Range"],
            y=[ci_hi - ci_low],
            base=[ci_low],
            marker_color="rgba(99, 179, 237, 0.6)",
            name="Range",
            width=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=["Confidence Range"],
            y=[pred],
            mode="markers",
            marker=dict(size=18, color="#F6AD55", symbol="diamond"),
            name="Point Estimate",
        ))
        fig.update_layout(
            height=250,
            yaxis_title="AED",
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            yaxis=dict(tickformat=",.0f"),
        )
    
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.info(f"Based on the inputs provided, this **{prop_type}** ({prop_subtype}) in **{area}** of **{size_m2} m²** with **{rooms} room(s)** is estimated at **{fmt_aed(pred)}**, with a confidence range of **{fmt_aed(ci_low)}** to **{fmt_aed(ci_hi)}**.")
        st.success("Prediction saved. Head to the **Explanation** page to see why the model gave this price.")