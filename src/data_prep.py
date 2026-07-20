"""
Week 1 - Step: Load and clean the two source datasets.

Diabetes  -> data/raw/diabetes_raw.csv     (Pima Indians Diabetes, UCI)
Cardio    -> data/raw/cardio_train.csv     (Cardiovascular Disease dataset, Kaggle)

Run:
    python src/data_prep.py
"""
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def load_diabetes():
    cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    df = pd.read_csv(f"{RAW_DIR}/diabetes_raw.csv", header=None, names=cols)

    # In this dataset, 0 is used as a stand-in for "missing" in columns
    # where a real zero is medically impossible. Treat those as NaN, then
    # impute with the median so we don't quietly bias the model with fake zeros.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_as_missing:
        df[c] = df[c].replace(0, np.nan)
        df[c] = df[c].fillna(df[c].median())

    return df


def load_cardio():
    df = pd.read_csv(f"{RAW_DIR}/cardio_train.csv", sep=";")

    # age is stored in days here -- convert to years to match how a
    # clinician (and your other dataset) would think about it
    df["age_years"] = (df["age"] / 365.25).round(1)

    # Real-world clinical data has entry errors: blood pressure of 16000,
    # negative values, systolic lower than diastolic, etc. Filter to
    # physiologically plausible ranges rather than silently keeping them.
    df = df[
        (df["ap_hi"] > 80) & (df["ap_hi"] < 250) &
        (df["ap_lo"] > 40) & (df["ap_lo"] < 200) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["height"] > 120) & (df["height"] < 220) &
        (df["weight"] > 30) & (df["weight"] < 200)
    ].copy()

    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(1)

    return df


if __name__ == "__main__":
    diabetes = load_diabetes()
    cardio = load_cardio()

    diabetes.to_csv(f"{OUT_DIR}/diabetes_clean.csv", index=False)
    cardio.to_csv(f"{OUT_DIR}/cardio_clean.csv", index=False)

    print("Diabetes dataset:", diabetes.shape, "-> saved to data/processed/diabetes_clean.csv")
    print(diabetes.head(3))
    print()
    print("Cardio dataset (after cleaning):", cardio.shape, "-> saved to data/processed/cardio_clean.csv")
    print(cardio[["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "cardio"]].head(3))
