"""
Week 2 - Step 4: SHAP explainability for the multi-output model.

MultiOutputClassifier trains one separate XGBoost model per disease
internally (model.estimators_[0] for the first target column, [1] for the
second) -- so SHAP has to be applied to each one separately, then presented
together as "here's what drives each disease's prediction."

Run:
    python src/shap_explain.py
"""
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no display in this environment -- save to file instead
import matplotlib.pyplot as plt

FEATURES = ["age", "bmi", "bp_diastolic", "glucose_mgdl", "cholesterol_cat"]

if __name__ == "__main__":
    model = joblib.load("models/multioutput_xgb.joblib")

    # y_train was [diabetes_label, cvd_label], so estimators_ follows that order.
    diabetes_model = model.estimators_[0]
    cvd_model = model.estimators_[1]

    diabetes_test = pd.read_csv("data/processed/diabetes_real_test.csv")
    cardio_test = pd.read_csv("data/processed/cardio_real_test.csv")

    print("=" * 70)
    print("SHAP -- Diabetes output, explained on real diabetes test patients")
    print("=" * 70)
    explainer_d = shap.TreeExplainer(diabetes_model)
    shap_values_d = explainer_d.shap_values(diabetes_test[FEATURES])
    mean_abs_d = pd.Series(
        abs(shap_values_d).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    print("Mean |SHAP value| per feature (higher = more influence on this prediction):")
    print(mean_abs_d.round(4))

    plt.figure()
    shap.summary_plot(shap_values_d, diabetes_test[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig("models/shap_diabetes_summary.png", dpi=120)
    plt.close()
    print("Saved plot -> models/shap_diabetes_summary.png\n")

    print("=" * 70)
    print("SHAP -- Cardiovascular output, explained on real cardio test patients")
    print("=" * 70)
    explainer_c = shap.TreeExplainer(cvd_model)
    shap_values_c = explainer_c.shap_values(cardio_test[FEATURES])
    mean_abs_c = pd.Series(
        abs(shap_values_c).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    print("Mean |SHAP value| per feature (higher = more influence on this prediction):")
    print(mean_abs_c.round(4))

    plt.figure()
    shap.summary_plot(shap_values_c, cardio_test[FEATURES], show=False)
    plt.tight_layout()
    plt.savefig("models/shap_cvd_summary.png", dpi=120)
    plt.close()
    print("Saved plot -> models/shap_cvd_summary.png")
