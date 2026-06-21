# 🏙️ Dubai Property AVM — Frontend

A Streamlit-based Automated Valuation Model (AVM) for Dubai real estate.
Built for CSCI 323 — Machine Learning, University of Wollongong Dubai.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone <repo-url>
cd Dubai-Real-Estate-ChatBot
```

### 2. Navigate to the Frontend folder
```bash
cd Frontend
```

### 3. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 📄 Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Landing page with model overview and accuracy snapshot |
| 🏢 Valuation | Property input form → price estimate with confidence range |
| 🔍 Explanation | SHAP-based breakdown of why the model gave that price |
| 📊 Market Trends | Charts and insights from the model's test predictions |
| ℹ️ About | Model details, accuracy, limitations, team info |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_API` | `True` | Set to `False` in `helpers.py` to use real backend |
| Backend URL | `http://localhost:8000` | Update `BACKEND_URL` in `helpers.py` |

---

## 🔌 Switching to Real Backend

When the backend teammate has the API running:

1. Open `helpers.py`
2. Change line 6 from:
```python
USE_MOCK_API = True
```
to:
```python
USE_MOCK_API = False
```
3. Make sure `BACKEND_URL` points to the correct address
4. Restart the Streamlit app

---

## 📁 Folder Structure

Frontend/
├── app.py                  ← Home page
├── helpers.py              ← API client, mock data, formatters
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── DESIGN_NOTES.md         ← Design rationale
├── pages/
│   ├── 1_Valuation.py      ← Property valuation form
│   ├── 2_Explanation.py    ← SHAP explanation page
│   ├── 3_Market_Trends.py  ← Market charts
│   └── 4_About.py          ← Model info and limitations
└── assets/
└── screenshots/        ← UI screenshots for report

---

## 📦 Dependencies

- `streamlit >= 1.30`
- `pandas >= 2.0`
- `plotly >= 5.18`
- `requests >= 2.31`
- `numpy >= 1.24`

---

## 👥 Team

CSCI 323 Group Project — University of Wollongong Dubai · 2025