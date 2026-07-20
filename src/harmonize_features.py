"""
Week 2 - Step 1: Harmonize both datasets onto one shared feature schema.

The two real datasets measure different things. To build ONE multi-output
model, we need a common feature schema both diseases can be scored on:

    age              -- years, exists in both
    bmi              -- exists in both
    bp_diastolic     -- exists in both (diabetes: BloodPressure, cardio: ap_lo)
    glucose_mgdl      -- diabetes has this directly; cardio only has a
                         1/2/3 category, mapped to representative mg/dL values
    cholesterol_cat  -- cardio has this directly (1/2/3); diabetes never
                         measured cholesterol at all, so it's imputed as
                         1 ("normal") -- an explicit, documented assumption,
                         not a hidden one.

Run:
    python src/harmonize_features.py
"""
import pandas as pd

# WHO-style rough glucose category -> representative mg/dL midpoint.
# This is an approximation (cardio only gives us a bucket, not a number)
# and should be stated as an assumption in Paper 1's Methodology section.
GLUCOSE_CATEGORY_TO_MGDL = {1: 90, 2: 160, 3: 220}


def harmonize_diabetes(df):
    out = pd.DataFrame({
        "age": df["Age"],
        "bmi": df["BMI"],
        "bp_diastolic": df["BloodPressure"],
        "glucose_mgdl": df["Glucose"],
        "cholesterol_cat": 1,  # not measured in this dataset -- assumed "normal"
        "target": df["Outcome"],
    })
    return out


def harmonize_cardio(df):
    out = pd.DataFrame({
        "age": df["age_years"],
        "bmi": df["bmi"],
        "bp_diastolic": df["ap_lo"],
        "glucose_mgdl": df["gluc"].map(GLUCOSE_CATEGORY_TO_MGDL),
        "cholesterol_cat": df["cholesterol"],
        "target": df["cardio"],
    })
    return out


if __name__ == "__main__":
    diabetes = pd.read_csv("data/processed/diabetes_clean.csv")
    cardio = pd.read_csv("data/processed/cardio_clean.csv")

    diabetes_h = harmonize_diabetes(diabetes)
    cardio_h = harmonize_cardio(cardio)

    diabetes_h.to_csv("data/processed/diabetes_harmonized.csv", index=False)
    cardio_h.to_csv("data/processed/cardio_harmonized.csv", index=False)

    print("Diabetes harmonized:", diabetes_h.shape)
    print(diabetes_h.head(3))
    print()
    print("Cardio harmonized:", cardio_h.shape)
    print(cardio_h.head(3))
    print()
    print("NOTE: cholesterol_cat is a real measurement for cardio patients,")
    print("but an assumed constant (1='normal') for diabetes patients, since")
    print("that dataset never measured cholesterol. This is a limitation to")
    print("state explicitly in Paper 1, not something to hide.")
