"""
Week 1 - Step: Baseline single-disease models (Logistic Regression, Random Forest).

These numbers are what you'll compare your Week 2 multi-output XGBoost model
against in Paper 1's Results section.

Run:
    python src/train_baseline.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, results):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    results.append(row)
    return row


def run_for_disease(disease_name, X, y):
    print("=" * 70)
    print(disease_name)
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    evaluate("Logistic Regression", log_reg, X_test_scaled, y_test, results)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    evaluate("Random Forest", rf, X_test, y_test, results)

    df_results = pd.DataFrame(results).set_index("model").round(3)
    print(df_results)
    print()
    return df_results


if __name__ == "__main__":
    diabetes = pd.read_csv("data/processed/diabetes_clean.csv")
    X_d = diabetes.drop(columns=["Outcome"])
    y_d = diabetes["Outcome"]
    diabetes_results = run_for_disease("DIABETES - Baseline Models", X_d, y_d)

    cardio = pd.read_csv("data/processed/cardio_clean.csv")
    feature_cols = ["age_years", "gender", "bmi", "ap_hi", "ap_lo",
                     "cholesterol", "gluc", "smoke", "alco", "active"]
    X_c = cardio[feature_cols]
    y_c = cardio["cardio"]
    cardio_results = run_for_disease("CARDIOVASCULAR - Baseline Models", X_c, y_c)

    print("Saved nothing yet on purpose -- these numbers just need to live in your")
    print("notes for now. You'll put them in a comparison table against your")
    print("Week 2 multi-output XGBoost model, and again in Paper 1's Results section.")
