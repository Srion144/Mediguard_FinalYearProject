const FIELD_LABELS = {
  age: 'Age',
  bmi: 'BMI',
  bp_diastolic: 'Diastolic BP',
  glucose_mgdl: 'Glucose',
  cholesterol_cat: 'Cholesterol',
}

// Shows each driver as a bar growing left (pushes risk down) or right
// (pushes risk up) from a center zero-line -- a small "tornado chart",
// which is the standard honest way to show signed SHAP contributions.
export default function DriverBars({ title, drivers }) {
  if (!drivers || drivers.length === 0) return null
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)), 0.001)

  return (
    <div className="mt-3 w-full">
      <p className="font-body text-[11px] uppercase tracking-wide text-muted mb-2">{title}</p>
      <div className="space-y-1.5">
        {drivers.map((d) => {
          const widthPct = (Math.abs(d.shap_value) / maxAbs) * 42
          const isPositive = d.shap_value >= 0
          return (
            <div key={d.feature} className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-right font-body text-muted">{FIELD_LABELS[d.feature] || d.feature}</span>
              <div className="relative flex-1 h-2.5 rounded bg-canvas">
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border" />
                <div
                  className="absolute top-0 bottom-0 rounded"
                  style={{
                    width: `${widthPct}%`,
                    left: isPositive ? '50%' : `${50 - widthPct}%`,
                    backgroundColor: isPositive ? 'var(--color-risk-high)' : 'var(--color-risk-low)',
                  }}
                />
              </div>
              <span className="w-12 shrink-0 font-mono text-muted">
                {d.shap_value > 0 ? '+' : ''}
                {d.shap_value.toFixed(2)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
