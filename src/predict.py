"""
Week 3 - Hand-off 1: a single, clean predict() function wrapping the
trained multi-output model, ready to be dropped into Track B's FastAPI
/predict endpoint.

Track B needs to know exactly two things about this file:
  1. What predict() expects as input (see validate_input's ranges below)
  2. What predict() returns (see the return dict in predict())
Both are also written out in INPUT_OUTPUT_SPEC.md alongside this file.

Run directly to sanity-check with sample patients:
    python src/predict.py
"""
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

# Loaded lazily so importing this module doesn't immediately load the
# model file -- useful for Track B's FastAPI app, which will import this
# once at startup rather than re-loading the model on every request.
_model = None
_explainer_diabetes = None
_explainer_cvd = None


def _load():
    global _model, _explainer_diabetes, _explainer_cvd
    if _model is None:
        _model = joblib.load("models/multioutput_xgb.joblib")
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
             (see VALID_RANGES above for expected units/ranges)

    Returns a dict -- see INPUT_OUTPUT_SPEC.md for the full contract.
    On invalid input, returns {"error": True, "messages": [...]} instead
    of raising an exception, so the API layer can turn this directly into
    a clean 400 response without needing a try/except around every call.
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


if __name__ == "__main__":
    samples = [
        {"age": 55, "bmi": 32.0, "bp_diastolic": 95, "glucose_mgdl": 190, "cholesterol_cat": 3},
        {"age": 28, "bmi": 21.5, "bp_diastolic": 70, "glucose_mgdl": 85, "cholesterol_cat": 1},
        {"age": 999, "bmi": 32.0, "bp_diastolic": 95, "glucose_mgdl": 190, "cholesterol_cat": 3},  # invalid: bad age
    ]
    for s in samples:
        print("Input: ", s)
        print("Output:", predict(s))
        print()
