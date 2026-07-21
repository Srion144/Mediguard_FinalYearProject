"""
This is Track A's src/predict.py, copied in and adapted for the backend.

ONE change from the original: the model file is now loaded relative to
THIS FILE's own location (using pathlib), instead of a path relative to
whatever folder you happened to launch the command from. This matters
because `uvicorn` might get started from the `backend/` folder, the repo
root, or anywhere else -- a plain "models/multioutput_xgb.joblib" string
would silently look in the wrong place depending on that. Everything else
(validation rules, output shape) is identical to INPUT_OUTPUT_SPEC.md.
"""
from pathlib import Path

import joblib
import pandas as pd
import shap

FEATURES = ["age", "bmi", "bp_diastolic", "glucose_mgdl", "cholesterol_cat"]

VALID_RANGES = {
    "age": (1, 120),
    "bmi": (10, 80),
    "bp_diastolic": (30, 150),
    "glucose_mgdl": (40, 400),
    "cholesterol_cat": (1, 3),  # 1=normal, 2=above normal, 3=well above normal
}

MODEL_PATH = Path(__file__).resolve().parent / "multioutput_xgb.joblib"

_model = None
_explainer_diabetes = None
_explainer_cvd = None


def _load():
    global _model, _explainer_diabetes, _explainer_cvd
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _explainer_diabetes = shap.TreeExplainer(_model.estimators_[0])
        _explainer_cvd = shap.TreeExplainer(_model.estimators_[1])
    return _model, _explainer_diabetes, _explainer_cvd


def validate_input(patient: dict):
    """Returns a list of human-readable error strings. Empty list = valid."""
    errors = []
    for f in FEATURES:
        if f not in patient:
            errors.append(f"Missing required field: {f}")
            continue
        val = patient[f]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            errors.append(f"'{f}' must be a number, got {type(val).__name__}")
            continue
        lo, hi = VALID_RANGES[f]
        if not (lo <= val <= hi):
            errors.append(f"'{f}'={val} is outside the plausible range [{lo}, {hi}]")
    return errors


def top_drivers(shap_vals, k=3):
    pairs = sorted(zip(FEATURES, shap_vals), key=lambda p: abs(p[1]), reverse=True)[:k]
    return [{"feature": f, "shap_value": round(float(v), 4)} for f, v in pairs]


def predict(patient: dict) -> dict:
    """
    patient: dict with keys age, bmi, bp_diastolic, glucose_mgdl, cholesterol_cat

    Returns a dict matching INPUT_OUTPUT_SPEC.md exactly. On invalid input,
    returns {"error": True, "messages": [...]} instead of raising -- so the
    FastAPI endpoint below can return this directly as a normal 200 response
    and let the frontend check result.error, rather than needing a
    try/except around every call.
    """
    errors = validate_input(patient)
    if errors:
        return {"error": True, "messages": errors}

    model, explainer_d, explainer_c = _load()
    X = pd.DataFrame([{f: patient[f] for f in FEATURES}])

    diabetes_prob = float(model.predict_proba(X)[0][:, 1][0])
    cvd_prob = float(model.predict_proba(X)[1][:, 1][0])

    shap_d = explainer_d.shap_values(X)[0]
    shap_c = explainer_c.shap_values(X)[0]

    return {
        "error": False,
        "diabetes_risk_probability": round(diabetes_prob, 4),
        "diabetes_risk_label": int(diabetes_prob >= 0.5),
        "cvd_risk_probability": round(cvd_prob, 4),
        "cvd_risk_label": int(cvd_prob >= 0.5),
        "top_diabetes_drivers": top_drivers(shap_d),
        "top_cvd_drivers": top_drivers(shap_c),
    }
