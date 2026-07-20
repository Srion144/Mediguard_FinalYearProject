# Week 2 — What Was Added

Drop these 4 files into your existing `src/` folder (from Week 1), then run
in this exact order:

```bash
python src/harmonize_features.py
python src/generate_synthetic_cohort.py
python src/train_multioutput.py
python src/shap_explain.py
```

## What each one does

1. **harmonize_features.py** — maps both datasets onto one shared 5-feature
   schema: age, bmi, bp_diastolic, glucose_mgdl, cholesterol_cat. Diabetes
   never measured cholesterol, so it's set to a constant assumed "normal"
   value for that dataset — an explicit, stated limitation.

2. **generate_synthetic_cohort.py** — splits off a REAL held-out test set
   for each disease FIRST (untouched afterward), trains a "teacher" Random
   Forest per disease on the shared features, then builds 6,000 synthetic
   patients by resampling real feature values and labeling them with both
   teachers. This is what turns two separate single-disease datasets into
   one genuine multi-output training set.

3. **train_multioutput.py** — trains one `MultiOutputClassifier(XGBClassifier())`
   on the synthetic cohort, then evaluates it on the REAL held-out test
   data saved in step 2 (not synthetic data — this is the number that
   actually matters).

4. **shap_explain.py** — SHAP explainability for each disease output
   separately, since MultiOutputClassifier trains one internal model per
   target. Saves two summary plots to `models/`.

## Results from this run

**Teacher models (5 shared features only, evaluated on real held-out data):**

| Disease | Accuracy | ROC-AUC |
|---|---|---|
| Diabetes | 0.740 | 0.814 |
| Cardiovascular | 0.642 | 0.696 |

**Final multi-output model (trained on synthetic cohort, evaluated on real held-out data):**

| Disease | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Diabetes | 0.643 | 0.493 | 0.630 | 0.553 | 0.748 |
| Cardiovascular | 0.707 | 0.723 | 0.664 | 0.692 | 0.760 |

**Comparison against Week 1's original full-feature baselines:**

| | Week 1 (full features) | Week 2 multi-output (shared features, synthetic-trained) |
|---|---|---|
| Diabetes Accuracy / ROC-AUC | 0.740 / 0.816 | 0.643 / 0.748 |
| Cardio Accuracy / ROC-AUC | 0.701 / 0.759 | 0.707 / 0.760 |

**Honest finding:** the multi-output model essentially matches Week 1's
baseline on cardiovascular risk, but underperforms it on diabetes. This is
a real, reportable tradeoff — training on synthetic labels (rather than
directly on real diabetes labels) costs some diabetes-specific accuracy in
exchange for the ability to predict both diseases jointly from one model.
This belongs in Paper 1's Discussion/Limitations section, not something to
hide.

**SHAP feature importance (mean |SHAP value|):**

| Feature | Diabetes output | Cardio output |
|---|---|---|
| glucose_mgdl | 3.92 (highest) | 0.08 |
| bmi | 1.29 | 0.30 |
| bp_diastolic | 1.06 | 0.83 (highest) |
| age | 0.80 | 0.41 |
| cholesterol_cat | 0.12 | 0.45 |

Both rankings are clinically sensible: glucose dominates diabetes risk,
blood pressure and cholesterol dominate cardiovascular risk — exactly what
a clinician would expect, which is a good sign the model learned something
real rather than noise.

## Next step (Week 3)

Wrap the trained model as a clean `predict()` function and hand it (plus
the input/output spec) to Track B — this is Hand-off 1 from your roadmap.
