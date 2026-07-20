"""
Week 1 - Step: Basic exploratory data analysis on both cleaned datasets.

Run:
    python src/eda.py
"""
import pandas as pd

pd.set_option("display.width", 120)


def summarize(name, df, target_col):
    print("=" * 70)
    print(f"{name}  |  shape={df.shape}")
    print("=" * 70)
    print("\nClass balance (target =", target_col, "):")
    print(df[target_col].value_counts(normalize=True).round(3))
    print("\nNumeric summary:")
    print(df.describe().round(2).T[["mean", "std", "min", "max"]])
    print()


if __name__ == "__main__":
    diabetes = pd.read_csv("data/processed/diabetes_clean.csv")
    cardio = pd.read_csv("data/processed/cardio_clean.csv")

    summarize("DIABETES (Pima)", diabetes, "Outcome")
    summarize("CARDIOVASCULAR", cardio, "cardio")

    print("=" * 70)
    print("Shared feature check (columns that exist, in comparable form, in BOTH datasets)")
    print("=" * 70)
    print("Diabetes has:  Glucose, BMI, BloodPressure(diastolic-only), Age")
    print("Cardio has:    gluc(categorical 1-3), bmi, ap_hi/ap_lo, age_years, cholesterol")
    print("-> Age, BMI, and a blood-pressure/glucose signal are the overlapping axes")
    print("   your multi-output model (Week 2) will be built around.")
