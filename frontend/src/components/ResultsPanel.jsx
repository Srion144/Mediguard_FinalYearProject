import RiskGauge from './RiskGauge'
import DriverBars from './DriverBars'

export default function ResultsPanel({ result, errorMessages }) {
  if (errorMessages) {
    return (
      <div className="bg-surface border border-border rounded-2xl p-6">
        <h2 className="font-display text-xl text-ink mb-2">Check the details</h2>
        <ul className="list-disc list-inside text-sm font-body space-y-1" style={{ color: 'var(--color-risk-high)' }}>
          {errorMessages.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-surface border border-border rounded-2xl p-6 flex items-center justify-center text-center min-h-[280px]">
        <p className="font-body text-sm text-muted max-w-xs">
          Fill in the patient&rsquo;s details and select &ldquo;Assess risk&rdquo; to see their diabetes and
          cardiovascular risk here.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-surface border border-border rounded-2xl p-6">
      <h2 className="font-display text-xl text-ink mb-4">Risk assessment</h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col items-center">
          <RiskGauge label="Diabetes" probability={result.diabetes_risk_probability} />
          <DriverBars title="What's driving this" drivers={result.top_diabetes_drivers} />
        </div>
        <div className="flex flex-col items-center">
          <RiskGauge label="Cardiovascular" probability={result.cvd_risk_probability} />
          <DriverBars title="What's driving this" drivers={result.top_cvd_drivers} />
        </div>
      </div>
    </div>
  )
}
