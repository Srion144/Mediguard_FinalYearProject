"""
Week 2 - Step 2: Train teacher models on the shared 5-feature schema, then
use them to build and label a synthetic cohort for multi-output training.

Why synthetic: the two real datasets are different patients, so there's no
way to directly train "one patient, two disease labels." Instead:

  1. Train a Random Forest per disease, using ONLY the shared features
     (so both speak the same input language).
  2. Generate synthetic patients by resampling real feature *values*
     (age, BMI, BP, glucose) from the real data -- so the synthetic
     population's individual feature distributions are grounded in
     reality, even though no single synthetic "patient" is a real person.
  3. Score every synthetic patient with BOTH teachers -> now each
     synthetic patient has a diabetes-risk label AND a CVD-risk label,
     which is what makes this a genuine multi-output dataset.

Run:
    python src/generate_synthetic_cohort.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

RANDOM_STATE = 42
N_SYNTHETIC = 6000
FEATURES = ["age", "bmi", "bp_diastolic", "glucose_mgdl", "cholesterol_cat"]

np.random.seed(RANDOM_STATE)


def train_teacher(name, df_train, df_test):
    X_train, y_train = df_train[FEATURES], df_train["target"]
    X_test, y_test = df_test[FEATURES], df_test["target"]

    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"{name} teacher (shared 5-feature schema only, evaluated on held-out real test set):")
    print(f"  Accuracy: {accuracy_score(y_test, preds):.3f}  ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    return model


def build_synthetic_cohort(diabetes_h, cardio_h, diabetes_teacher, cardio_teacher):
    # Pool age/bmi/bp_diastolic/glucose_mgdl from BOTH datasets -- these are
    # real measurements from real patients in both source datasets, so
    # pooling them keeps the synthetic population's marginal distributions
    # grounded in reality rather than invented from scratch.
    pooled = pd.concat([
        diabetes_h[["age", "bmi", "bp_diastolic", "glucose_mgdl"]],
        cardio_h[["age", "bmi", "bp_diastolic", "glucose_mgdl"]],
    ], ignore_index=True)

    synthetic = pooled.sample(n=N_SYNTHETIC, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)

    # cholesterol_cat is only a real measurement in the cardio dataset,
    # so it's resampled from THAT real distribution only.
    synthetic["cholesterol_cat"] = cardio_h["cholesterol_cat"].sample(
        n=N_SYNTHETIC, replace=True, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Score every synthetic patient with both teachers.
    synthetic["diabetes_risk_prob"] = diabetes_teacher.predict_proba(synthetic[FEATURES])[:, 1]
    synthetic["cvd_risk_prob"] = cardio_teacher.predict_proba(synthetic[FEATURES])[:, 1]
    synthetic["diabetes_label"] = (synthetic["diabetes_risk_prob"] >= 0.5).astype(int)
    synthetic["cvd_label"] = (synthetic["cvd_risk_prob"] >= 0.5).astype(int)

    return synthetic


if __name__ == "__main__":
    diabetes_h = pd.read_csv("data/processed/diabetes_harmonized.csv")
    cardio_h = pd.read_csv("data/processed/cardio_harmonized.csv")

    # Hold out a real test set for EACH disease before anything else touches
    # the data. This test set is never used for teacher training or for the
    # synthetic resampling pool -- it stays untouched until Step 3, where the
    # final multi-output model gets evaluated against it.
    diabetes_train, diabetes_test = train_test_split(
        diabetes_h, test_size=0.2, random_state=RANDOM_STATE, stratify=diabetes_h["target"]
    )
    cardio_train, cardio_test = train_test_split(
        cardio_h, test_size=0.2, random_state=RANDOM_STATE, stratify=cardio_h["target"]
    )
    diabetes_test.to_csv("data/processed/diabetes_real_test.csv", index=False)
    cardio_test.to_csv("data/processed/cardio_real_test.csv", index=False)

    print("=" * 70)
    print("Training teacher models (shared-feature-only baselines)")
    print("=" * 70)
    diabetes_teacher = train_teacher("Diabetes", diabetes_train, diabetes_test)
    cardio_teacher = train_teacher("Cardiovascular", cardio_train, cardio_test)
    print()
    print("(Compare these to Week 1's full-feature baselines -- some accuracy")
    print(" is expected to drop, since we're now using only 5 shared features")
    print(" instead of each dataset's full original feature set.)")
    print()

    print("=" * 70)
    print(f"Building synthetic cohort (n={N_SYNTHETIC}) from TRAIN portions only")
    print("=" * 70)
    synthetic = build_synthetic_cohort(diabetes_train, cardio_train, diabetes_teacher, cardio_teacher)
    synthetic.to_csv("data/processed/synthetic_cohort.csv", index=False)

    print(synthetic.head(5))
    print()
    print("Label distribution in synthetic cohort:")
    print("  Diabetes positive rate:", round(synthetic["diabetes_label"].mean(), 3))
    print("  CVD positive rate:     ", round(synthetic["cvd_label"].mean(), 3))
    print("  Both positive (comorbid):", round(
        ((synthetic["diabetes_label"] == 1) & (synthetic["cvd_label"] == 1)).mean(), 3
    ))
    print()
    print("Saved to data/processed/synthetic_cohort.csv")
    print("Real held-out test sets saved to diabetes_real_test.csv / cardio_real_test.csv")
    print("(untouched by any training so far -- reserved for Step 3's final evaluation)")
