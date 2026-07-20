"""
Week 2 - Step 3: Train ONE multi-output XGBoost model on the synthetic
cohort, then evaluate it on the REAL held-out test data from both diseases
(not synthetic data) -- this is the number that actually matters and the
one that goes in Paper 1.

Run:
    python src/train_multioutput.py
"""
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = ["age", "bmi", "bp_diastolic", "glucose_mgdl", "cholesterol_cat"]
RANDOM_STATE = 42


def evaluate_single_output(name, y_true, probs):
    preds = (probs >= 0.5).astype(int)
    print(f"  {name}:")
    print(f"    Accuracy:  {accuracy_score(y_true, preds):.3f}")
    print(f"    Precision: {precision_score(y_true, preds, zero_division=0):.3f}")
    print(f"    Recall:    {recall_score(y_true, preds, zero_division=0):.3f}")
    print(f"    F1:        {f1_score(y_true, preds, zero_division=0):.3f}")
    print(f"    ROC-AUC:   {roc_auc_score(y_true, probs):.3f}")


if __name__ == "__main__":
    synthetic = pd.read_csv("data/processed/synthetic_cohort.csv")
    X_train = synthetic[FEATURES]
    y_train = synthetic[["diabetes_label", "cvd_label"]]

    print("=" * 70)
    print("Training multi-output XGBoost on synthetic cohort")
    print("=" * 70)
    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
    )
    model.fit(X_train, y_train)
    print("Trained on", X_train.shape[0], "synthetic patients.\n")

    # --- Evaluate on REAL held-out test data (the number that matters) ---
    diabetes_test = pd.read_csv("data/processed/diabetes_real_test.csv")
    cardio_test = pd.read_csv("data/processed/cardio_real_test.csv")

    print("=" * 70)
    print("Evaluating on REAL held-out test data (never seen during training)")
    print("=" * 70)

    diabetes_probs = model.predict_proba(diabetes_test[FEATURES])[0][:, 1]
    evaluate_single_output("Diabetes (real test set)", diabetes_test["target"], diabetes_probs)
    print()
    cvd_probs = model.predict_proba(cardio_test[FEATURES])[1][:, 1]
    evaluate_single_output("Cardiovascular (real test set)", cardio_test["target"], cvd_probs)

    # Save the model + a small sample for the SHAP step next.
    import joblib
    joblib.dump(model, "models/multioutput_xgb.joblib")
    print("\nModel saved to models/multioutput_xgb.joblib")
