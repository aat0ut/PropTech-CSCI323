# Dubai Real Estate AVM (Proptech)

> An interpretable Automated Valuation Model for Dubai's real estate market, built on Dubai Land Department transaction data.

## Overview

The following project replaces vague, weak natural-language valuation tools with a more transparent data-driven pipeline. 

It combines:

- A stacking regressor (tree-based) which is trained on structured features attained from Dubai Land Department's transactional data.
- An interpretability layer using SHAP which is rooted in cooperative game theory, transforming any individual property valuations into properly auditable outputs.

This project's goal is to provide PropTech stakeholders not only with price estimations, but also an explanation for why that estimate was made, all while addressing the trust gap in the algorithm which limits AVM adoption.

## Key Features

- Feature engineering from CSV files from the DLD.
- Tree-based stacking regressor (base learners + meta-learner).
- Multi-level model evaluation.
- SHAP-based interpretability for  global feature importance and single predictions.

## Tech Stack

- Language: Python
- Environment: Jupyter Notebooks (.ipynb)
- Core libraries: pandas, numpy, xgboost, lightgbm, sklearn, warnings, joblib, json, os

## Dataset

- Source: Dubai Land Department (DLD)
- Format: CSV


## Methodology

1. Feature engineering — temporal and spatial features are created from raw transactions (e.g. location-based encodings, time-based aggregates).
2. Encoding & preprocessing — feature columns, categorical encodings, and global means are computed and saved as reusable artifacts (see `Artifacts` below).
3. Model training — a stacking regressor is trained as base learners.
4. Evaluation — performance of models is compared against a baseline and deconstructed by test year and by property type  (see `Results`).
5. Explainability — SHAP values are calculated based on the trained stack model explaining individual predictions and global feature importance.

### Artifacts produced (column names are at the top of this list)

| Artifact | Format | Purpose |
| Encoding maps | `.json` | Maps categorical values to encoded representations |
| Feature columns | `.json` | Canonical list/order of model input features |
| Global means | `.json` | Fallback/imputation values used during preprocessing |
| Trained models | `.pkl` | Serialized base learners and stacking regressor |

## Results

Evaluation metrics are stored in the `results/` folder <!-- TODO: confirm exact filenames -->, covering:

- Baseline metrics — performance of a simple baseline model
- Baseline vs. stacked — direct comparison showing the lift from stacking
- Property type metrics — performance broken down by property category (e.g. apartment, villa, townhouse)
- Test year metrics — performance broken down by held-out test year

## Contributors

- Abdallah Atout
- Albaraa 
- Karim Zaripov
- Rena Khalilova
- Seyed Erfan Alavian