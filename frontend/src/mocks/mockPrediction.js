// Week 2 mock -- shaped EXACTLY like INPUT_OUTPUT_SPEC.md's real output,
// so every component that consumes it (RiskGauge, DriverBars, ResultsPanel)
// needs zero changes when Week 3 swaps this for the real backend call.
//
// The numbers below are NOT the real model -- just rough, made-up rules so
// the UI reacts to whatever the user types, which is all a mock needs to do.
export function mockPrediction(patient) {
  const glucoseFactor = clamp01((patient.glucose_mgdl - 70) / 250)
  const bpFactor = clamp01((patient.bp_diastolic - 60) / 70)

  return {
    error: false,
    diabetes_risk_probability: round4(glucoseFactor),
    diabetes_risk_label: glucoseFactor >= 0.5 ? 1 : 0,
    cvd_risk_probability: round4(bpFactor),
    cvd_risk_label: bpFactor >= 0.5 ? 1 : 0,
    top_diabetes_drivers: [
      { feature: 'glucose_mgdl', shap_value: 2.1 },
      { feature: 'bmi', shap_value: 0.8 },
      { feature: 'age', shap_value: 0.3 },
    ],
    top_cvd_drivers: [
      { feature: 'bp_diastolic', shap_value: 1.4 },
      { feature: 'cholesterol_cat', shap_value: 0.5 },
      { feature: 'age', shap_value: 0.2 },
    ],
  }
}

function clamp01(n) {
  return Math.min(Math.max(n, 0), 1)
}

function round4(n) {
  return Math.round(n * 10000) / 10000
}
