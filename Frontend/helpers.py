import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

BASE_DIR          = os.path.dirname(__file__)
MODELING_DIR      = os.path.join(BASE_DIR, "..", "Modeling")
ENCODING_MAPS     = os.path.join(MODELING_DIR, "models", "encoding_maps.json")
TEST_PREDICTIONS  = os.path.join(MODELING_DIR, "results", "test_predictions.csv")
WATERFALL_JSON    = os.path.join(MODELING_DIR, "results", "xai", "waterfall_arrays.json")
FEATURE_COL_JSON  = os.path.join(MODELING_DIR, "models", "feature_columns.json")
GLOBAL_MEANS_JSON = os.path.join(MODELING_DIR, "models", "global_means.json")

# ── Load encoding maps ────────────────────────────────────────────────────────
def load_encoding_maps():
    try:
        with open(ENCODING_MAPS, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "area":             {"Dubai Marina": 0, "Downtown Dubai": 1},
            "project":          {"Marina Heights": 0},
            "master_project":   {"Dubai Marina": 0},
            "nearest_metro":    {"DAMAC Properties Metro Station": 0},
            "nearest_mall":     {"Marina Mall": 0},
            "nearest_landmark": {"Burj Al Arab": 0},
            "prop_sb_type":     {"Flat": 0, "Studio": 1},
        }

def load_global_means():
    try:
        with open(GLOBAL_MEANS_JSON, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"price_per_sqm_mean": 19502.97, "fallback_for_unseen_category": 19502.97}

def load_feature_columns():
    try:
        with open(FEATURE_COL_JSON, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

MAPS         = load_encoding_maps()
GLOBAL_MEANS = load_global_means()
FEATURE_COLS = load_feature_columns()

area_options     = sorted(MAPS.get("area", {}).keys())
project_options  = sorted(MAPS.get("project", {}).keys())
master_options   = sorted(MAPS.get("master_project", {}).keys())
metro_options    = sorted(MAPS.get("nearest_metro", {}).keys())
mall_options     = sorted(MAPS.get("nearest_mall", {}).keys())
landmark_options = sorted(MAPS.get("nearest_landmark", {}).keys())

PROP_TYPES    = ["Unit", "Villa", "Land", "Building"]
PROP_SUBTYPES = sorted(MAPS.get("prop_sb_type", {"Flat": 0, "Studio": 1}).keys())
USAGES        = ["Residential", "Commercial", "Hospitality", "Industrial"]

# ── Load ML models directly ───────────────────────────────────────────────────
def _load_models():
    if not JOBLIB_AVAILABLE:
        return None, None, None
    try:
        models = {
            "xgb":  joblib.load(os.path.join(MODELING_DIR, "models", "xgb.pkl")),
            "lgbm": joblib.load(os.path.join(MODELING_DIR, "models", "lgbm.pkl")),
            "rf":   joblib.load(os.path.join(MODELING_DIR, "models", "rf.pkl")),
        }
        ridge  = joblib.load(os.path.join(MODELING_DIR, "models", "ridge.pkl"))
        scaler = joblib.load(os.path.join(MODELING_DIR, "models", "meta_scaler.pkl"))
        print("✅ Real models loaded successfully!")
        return models, ridge, scaler
    except Exception as e:
        print(f"⚠️ Could not load models: {e} — using mock")
        return None, None, None

MODELS, RIDGE, META_SCALER = _load_models()
USE_REAL_MODELS = MODELS is not None

# ── Feature encoding ──────────────────────────────────────────────────────────
def encode_features(features: dict) -> pd.DataFrame:
    now      = datetime.now()
    month    = now.month
    year     = now.year
    fallback = GLOBAL_MEANS.get("fallback_for_unseen_category", 19502.97)

    def cat_encode(field, val):
        return MAPS.get(field, {}).get(val, fallback)

    prop_type  = features.get("prop_type", "Unit")
    usage      = features.get("usage", "Residential")
    is_offplan = int(features.get("is_offplan", 0))

    # One-hot usage
    usage_col_map = {
        "Commercial":  "usage_Commercial",
        "Hospitality": "usage_Hospitality_",
        "Industrial":  "usage_Industrial",
        "Residential": "usage_Residential",
    }
    usage_cols = {col: 0 for col in [
        "usage_Commercial", "usage_Hospitality_", "usage_Industrial",
        "usage_Multi_Use", "usage_Other", "usage_Residential", "usage_Storage"
    ]}
    mapped = usage_col_map.get(usage)
    if mapped:
        usage_cols[mapped] = 1
    else:
        usage_cols["usage_Other"] = 1

    # Procedure — off-plan → Pre-registration, else Sell
    procedure_cols = {
        "procedure_Other":                  0,
        "procedure_Sell":                   0,
        "procedure_Sell___Pre_registration": 0,
        "procedure_Sell_Development":        0,
    }
    if is_offplan:
        procedure_cols["procedure_Sell___Pre_registration"] = 1
    else:
        procedure_cols["procedure_Sell"] = 1

    row = {
        "prop_sb_type":            cat_encode("prop_sb_type", features.get("prop_sb_type", "")),
        "is_offplan":              is_offplan,
        "area":                    cat_encode("area", features.get("area", "")),
        "project":                 cat_encode("project", features.get("project", "")),
        "master_project":          cat_encode("master_project", features.get("master_project", "")),
        "nearest_landmark":        cat_encode("nearest_landmark", features.get("nearest_landmark", "")),
        "nearest_metro":           cat_encode("nearest_metro", features.get("nearest_metro", "")),
        "nearest_mall":            cat_encode("nearest_mall", features.get("nearest_mall", "")),
        "rooms":                   int(features.get("rooms", 2)),
        "has_parking":             int(features.get("has_parking", 0)),
        "procedure_area":          float(features.get("procedure_area", 85)),
        "date_year":               year,
        "month_sin":               float(np.sin(2 * np.pi * month / 12)),
        "month_cos":               float(np.cos(2 * np.pi * month / 12)),
        "community_30d_avg_price": GLOBAL_MEANS.get("price_per_sqm_mean", 19502.97),
        "off_plan_vol_ratio":      0.4 if is_offplan else 0.1,
        "prop_type_Land":          1 if prop_type == "Land" else 0,
        "prop_type_Unit":          1 if prop_type == "Unit" else 0,
        "prop_type_Villa":         1 if prop_type == "Villa" else 0,
        **usage_cols,
        **procedure_cols,
    }

    return pd.DataFrame([row])[FEATURE_COLS]

# ── Real prediction ───────────────────────────────────────────────────────────
def _predict_real(features: dict) -> dict:
    X         = encode_features(features)
    xgb_pred  = float(MODELS["xgb"].predict(X)[0])
    lgbm_pred = float(MODELS["lgbm"].predict(X)[0])
    rf_pred   = float(MODELS["rf"].predict(X)[0])
    base      = np.array([[xgb_pred, lgbm_pred, rf_pred]])
    scaled    = META_SCALER.transform(base)
    final     = float(RIDGE.predict(scaled)[0])
    spread    = float(np.std([xgb_pred, lgbm_pred, rf_pred]))
    return {
        "predicted_value_aed": final,
        "ci_low":              max(0, final - 1.5 * spread),
        "ci_high":             final + 1.5 * spread,
        "base_spread":         spread,
        "model_version":       "ensemble_v1 (real)",
    }

# ── Load test predictions ─────────────────────────────────────────────────────
def _load_test_data():
    try:
        return pd.read_csv(TEST_PREDICTIONS)
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n   = 500
        predicted = rng.uniform(500_000, 5_000_000, n)
        spread    = rng.uniform(50_000, 200_000, n)
        return pd.DataFrame({
            "actual":      predicted * rng.uniform(0.88, 1.12, n),
            "predicted":   predicted,
            "ci_low":      predicted - spread,
            "ci_high":     predicted + spread,
            "base_spread": spread,
            "xgb_pred":    predicted * rng.uniform(0.95, 1.05, n),
            "lgbm_pred":   predicted * rng.uniform(0.95, 1.05, n),
            "rf_pred":     predicted * rng.uniform(0.95, 1.05, n),
        })

TEST_DATA = _load_test_data()

# ── Load real SHAP data ───────────────────────────────────────────────────────
def _load_waterfall_data():
    try:
        with open(WATERFALL_JSON, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

WATERFALL_DATA = _load_waterfall_data()

def _real_shap():
    if WATERFALL_DATA is None:
        return None
    import random
    row = random.choice(WATERFALL_DATA["rows"])
    return {
        "shap_values": {step["feature"]: step["shap_value"] for step in row["waterfall"]},
        "base_value":  WATERFALL_DATA["base_value"],
    }

def _mock_shap(features, predicted_value):
    rng = np.random.default_rng(abs(hash(str(features))) % (2**31))
    feature_names = ["area", "procedure_area", "prop_type_Unit", "rooms",
                     "nearest_metro", "is_offplan", "has_parking", "usage_Residential"]
    raw   = rng.uniform(-1, 1, len(feature_names))
    raw  /= raw.sum()
    delta = predicted_value - 1_200_000
    return {
        "shap_values": {k: round(float(v * delta), 0) for k, v in zip(feature_names, raw)},
        "base_value":  1_200_000,
    }

# ── Public functions ──────────────────────────────────────────────────────────
def predict_property(features: dict) -> dict:
    if USE_REAL_MODELS:
        try:
            return _predict_real(features)
        except Exception as e:
            print(f"Real prediction failed: {e} — falling back to mock")
    row = TEST_DATA.sample(1).iloc[0]
    return {
        "predicted_value_aed": float(row["predicted"]),
        "ci_low":              float(row["ci_low"]),
        "ci_high":             float(row["ci_high"]),
        "base_spread":         float(row["base_spread"]),
        "model_version":       "ensemble_v1 (mock)",
    }

def explain_property(features: dict, predicted_value: float) -> dict:
    real = _real_shap()
    if real is not None:
        return real
    return _mock_shap(features, predicted_value)

def fmt_aed(value: float) -> str:
    return f"{value:,.0f} AED"

def fmt_aed_compact(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M AED"
    if value >= 1_000:
        return f"{value/1_000:.1f}K AED"
    return fmt_aed(value)