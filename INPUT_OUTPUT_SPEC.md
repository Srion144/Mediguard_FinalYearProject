# MediGuard AI — predict() Input/Output Spec
### Hand-off 1: Track A → Track B

This is the contract for wiring `src/predict.py` into your FastAPI
`/predict` endpoint. Everything you need to know is below — you should
not need to look at the model internals at all.

## How to use it

```python
from src.predict import predict

result = predict({
    "age": 55,
    "bmi": 32.0,
    "bp_diastolic": 95,
    "glucose_mgdl": 190,
    "cholesterol_cat": 3
})
```

Drop this straight into your FastAPI endpoint — `predict()` takes a plain
dict and returns a plain dict (no custom objects), so it maps directly
onto a Pydantic request/response model.

## Input format

| Field | Type | Valid range | Notes |
|---|---|---|---|
| `age` | number | 1–120 | years |
| `bmi` | number | 10–80 | kg/m² |
| `bp_diastolic` | number | 30–150 | mmHg |
| `glucose_mgdl` | number | 40–400 | mg/dL |
| `cholesterol_cat` | number | 1, 2, or 3 | 1=normal, 2=above normal, 3=well above normal |

All five fields are required on every call. If your frontend form
collects cholesterol as a real mg/dL number instead of a category, map
it before calling `predict()`: roughly <200→1, 200–239→2, ≥240→3.

## Output format — success case

```json
{
  "error": false,
  "diabetes_risk_probability": 0.9999,
  "diabetes_risk_label": 1,
  "cvd_risk_probability": 0.6652,
  "cvd_risk_label": 1,
  "top_diabetes_drivers": [
    {"feature": "glucose_mgdl", "shap_value": 9.7621},
    {"feature": "age", "shap_value": 1.3291},
    {"feature": "cholesterol_cat", "shap_value": -0.246}
  ],
  "top_cvd_drivers": [
    {"feature": "bp_diastolic", "shap_value": 1.3213},
    {"feature": "glucose_mgdl", "shap_value": -0.9688},
    {"feature": "cholesterol_cat", "shap_value": 0.4772}
  ]
}
```

- `*_risk_probability` — a number from 0 to 1. Use this for the dashboard's
  risk visualization (e.g. a gauge or percentage bar), not just the label.
- `*_risk_label` — 0 or 1, using a 0.5 threshold. Use this for simple
  "at risk / not at risk" badges.
- `top_*_drivers` — the 3 features that most influenced this specific
  patient's prediction, ranked by SHAP magnitude. Positive `shap_value`
  pushes risk up, negative pushes it down. This is what should feed the
  "why am I seeing this result" explanation in the dashboard.

## Output format — invalid input case

```json
{
  "error": true,
  "messages": ["'age'=999 is outside the plausible range [1, 120]"]
}
```

`predict()` never raises an exception for bad input — it always returns
a dict. Check `result["error"]` first; if `true`, show `result["messages"]`
to the user (e.g. as form validation errors) instead of calling the model.

## What NOT to worry about

- You don't need to load the model yourself — `predict()` handles that
  internally (and only loads it once, on first call, not on every request).
- You don't need to know anything about XGBoost, SHAP, or how the model
  was trained — that's all encapsulated.
- You don't need to handle missing/malformed fields yourself — `predict()`
  validates everything and returns clean error messages.

## One thing to flag if it comes up

`cholesterol_cat` is a real measurement for cardio-style inputs but was an
assumed constant during part of training (see Week 2 log) — this doesn't
affect how you call `predict()`, but if a reviewer or your supervisor asks
about it, it's a known, documented modeling limitation, not a bug.
