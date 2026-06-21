# Design Notes — Dubai Property AVM Frontend

CSCI 323 · University of Wollongong Dubai · 2025

---

## 1. Target User Persona

The primary persona is a **semi-informed property buyer or real estate agent** in Dubai.
They understand what a property is worth roughly, but lack the data or tools to validate 
a price quickly. They are comfortable using web apps on both desktop and mobile, 
but are not data scientists — they should never see a raw number without context.

A secondary persona is a **property investor** comparing multiple areas or property 
types, who values the Market Trends page for aggregate insights.

---

## 2. Why These Input Fields

The input fields were dictated directly by the model's feature set, sourced from 
`encoding_maps.json`. Every dropdown maps to a feature the model was trained on — 
free-text input was deliberately avoided because the model has never seen spelling 
variations or abbreviations, and would produce garbage predictions.

Key decisions:
- **Dropdowns over free text** for area, project, metro, mall, landmark — enforces valid input
- **Number inputs with bounds** for size (20–5000 m²) and rooms (0–10) — catches outliers before they reach the model
- **Checkboxes** for parking and off-plan — binary features need binary inputs
- **3-column layout** — balances density and readability on desktop without scrolling

---

## 3. Why Always Show Confidence Intervals

The model's MAPE is ~13%. For a 2,000,000 AED property that is a ±260,000 AED 
uncertainty band. Displaying only a point estimate would imply a precision the 
model does not have, which is misleading to users making financial decisions.

The confidence range bar chart was chosen over a simple text display because 
a visual range is intuitively understood — users immediately grasp "the real 
value is probably somewhere in this blue band."

---

## 4. Explanation Page Design Decisions

The SHAP waterfall chart was chosen over a table because:
- Direction (up/down) is immediately visible via bar direction and color
- Magnitude is proportional to bar length — no mental math required
- The orange "Final Price" bar anchors the whole chart to a real number

Features were capped at **top 7** (doc recommendation) to avoid information overload. 
The "Other Factors" bucket preserves mathematical correctness without cluttering the chart.

Color choices:
- Green (#68D391) for positive contributions — universally understood as "good/up"
- Red (#FC8181) for negative contributions — universally understood as "bad/down"  
- Orange (#F6AD55) for the final price — distinct from both, neutral

The plain English paragraph below the chart was added because non-technical users 
may not immediately read a waterfall chart correctly. It bridges the visual and verbal.

---

## 5. Layouts Considered and Rejected

**Single-column layout for Valuation form** — rejected because 13 input fields 
in a single column requires excessive scrolling and feels like a government form.
The 3-column layout keeps everything visible above the fold on a standard laptop.

**Raw SHAP table as primary explanation** — rejected because a table of feature 
names and decimal values is meaningless to a non-technical user. The waterfall 
chart was always the primary; the table was demoted to a collapsible expander.

**Separate pages for each chart on Market Trends** — rejected as unnecessary 
navigation overhead. All charts on one scrollable page is appropriate for 
exploratory analysis.

---

## 6. Accessibility Considerations

- Color-blind safe: green/red pair was chosen with sufficient luminance contrast. 
  Bar direction (left vs right) also encodes direction, so color is not the only signal.
- All charts include axis labels, titles, and captions.
- Streamlit's default single-column mobile layout is preserved — `st.columns()` 
  is used only where content genuinely benefits from side-by-side display.
- Error messages are plain English — no stack traces are ever shown to users.

---

## 7. What We Would Change With More Time

- **Real encoding maps** — with `encoding_maps.json` connected, dropdowns would 
  show all 250+ real Dubai communities instead of the mock fallback list.
- **Map visualization** — plotting the selected area on an interactive Dubai map 
  would significantly improve spatial understanding.
- **Price history chart** — showing how prices in the selected area have trended 
  over the past 3 years would add powerful context.
- **Comparison mode** — letting users compare two properties side by side.
- **Mobile-optimized form** — the 3-column layout collapses to single-column on 
  mobile, which is functional but not polished. A dedicated mobile layout would help.
- **Real SHAP values** — replacing mock SHAP with real values from the 
  explainability teammate would make the Explanation page genuinely meaningful.

---

*Frontend built with Streamlit · Plotly · Pandas · Python*