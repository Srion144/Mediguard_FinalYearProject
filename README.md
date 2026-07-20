# MediGuard AI — Track A Starter (Week 1)

This is a working, tested starting point for the ML side of the project:
data loading, cleaning, EDA, and baseline models for both diseases.
Everything in here has already been run once and confirmed to work.

## What's inside

```
mediguard_starter/
├── data/
│   ├── raw/               <- original downloaded datasets (already included)
│   └── processed/         <- cleaned versions, produced by data_prep.py
├── src/
│   ├── data_prep.py       <- Step 1: load + clean both datasets
│   ├── eda.py              <- Step 2: explore both datasets
│   └── train_baseline.py   <- Step 3: baseline Logistic Regression + Random Forest
├── models/                 <- (empty for now — Week 2's saved model goes here)
├── notebooks/               <- (empty — use for any ad-hoc exploration)
└── requirements.txt
```

## How to run this on your own machine

### 1. Install Python (if you don't already have it)
Get Python 3.10+ from [python.org](https://www.python.org/downloads/). Confirm it worked:
```bash
python3 --version
```

### 2. Set up a virtual environment
This keeps this project's packages separate from everything else on your machine.

```bash
cd mediguard_starter
python3 -m venv venv

# activate it:
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

You'll know it worked because your terminal prompt will show `(venv)` at the start.

### 3. Install the packages
```bash
pip install -r requirements.txt
```

### 4. Run the three scripts, in order
```bash
python src/data_prep.py       # cleans both datasets, saves to data/processed/
python src/eda.py             # prints dataset summaries
python src/train_baseline.py  # trains + evaluates baseline models
```

## What each script actually does

**`data_prep.py`**
- Diabetes dataset: several columns use `0` to mean "missing" (a blood glucose
  of 0 isn't medically possible) — these are converted to NaN and filled with
  the column median rather than left as fake zeros.
- Cardio dataset: age is stored in days, converted to years. Rows with
  impossible blood pressure or height/weight values are filtered out
  (about 2% of rows) rather than silently kept.

**`eda.py`**
- Prints class balance and summary statistics for both datasets.
- Diabetes is 65/35 (non-diabetic/diabetic); Cardio is nearly 50/50 —
  both balanced enough that you don't need special imbalance handling yet.

**`train_baseline.py`**
- Trains Logistic Regression and Random Forest separately for each disease.
- These are the numbers you'll compare your Week 2 multi-output XGBoost
  model against, and the numbers that go in Paper 1's Results section.

## Results from this run (baseline, single-disease models)

**Diabetes:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.816 |

**Cardiovascular:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.731 | 0.758 | 0.669 | 0.711 | 0.793 |
| Random Forest | 0.701 | 0.699 | 0.697 | 0.698 | 0.759 |

Random Forest edges out Logistic Regression on diabetes; Logistic Regression
is actually slightly ahead on cardio. Both are legitimate baselines to beat.

## Next step (Week 2)

Build the multi-output XGBoost model that predicts both risks from one
model, add SHAP explainability, and compare its numbers against this table.
See your Track A Roadmap document for the full breakdown.

## A note on the data

- **Diabetes**: Pima Indians Diabetes dataset (UCI Machine Learning Repository).
- **Cardiovascular**: Cardiovascular Disease dataset, originally from Kaggle
  (sulianova/cardiovascular-disease-dataset).

Both are public research datasets — fine for prototyping. Your synopsis calls
for privacy-safe *synthetic* data generated from real distributions for the
final system; that's a Week 2 task once the real baseline is established.
