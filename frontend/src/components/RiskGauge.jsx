// A hand-drawn semicircular gauge (not a chart-library donut) -- meant to
// read like a clinical dial: risk fills the arc, coloured by band, with
// the percentage printed in the center in the mono/data typeface.

function polarToCartesian(cx, cy, r, angleDeg) {
  const angleRad = (angleDeg * Math.PI) / 180
  return {
    x: cx + r * Math.cos(angleRad),
    y: cy - r * Math.sin(angleRad),
  }
}

// Describes an SVG arc path from startAngle to endAngle (in degrees,
// measured like a compass: 180 = left, 90 = top, 0 = right), sweeping
// clockwise across the top -- i.e. exactly the upper half of a circle.
function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, startAngle)
  const end = polarToCartesian(cx, cy, r, endAngle)
  const largeArcFlag = startAngle - endAngle > 180 ? 1 : 0
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`
}

function riskBand(probability) {
  if (probability < 0.33) return { label: 'Low risk', color: 'var(--color-risk-low)' }
  if (probability < 0.66) return { label: 'Moderate risk', color: 'var(--color-risk-mid)' }
  return { label: 'High risk', color: 'var(--color-risk-high)' }
}

export default function RiskGauge({ label, probability }) {
  const size = 200
  const r = 78
  const cx = size / 2
  const cy = 108
  const clamped = Math.min(Math.max(probability ?? 0, 0), 1)
  const band = riskBand(clamped)

  const trackPath = describeArc(cx, cy, r, 180, 0)
  const fillPath = describeArc(cx, cy, r, 180, 180 - 180 * clamped)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={128} viewBox={`0 0 ${size} 128`} role="img" aria-label={`${label} risk: ${Math.round(clamped * 100)} percent, ${band.label}`}>
        <path d={trackPath} fill="none" stroke="var(--color-border)" strokeWidth="14" strokeLinecap="round" />
        <path d={fillPath} fill="none" stroke={band.color} strokeWidth="14" strokeLinecap="round" />
        <text
          x={cx}
          y={cy - 14}
          textAnchor="middle"
          style={{ fontFamily: 'var(--font-mono)', fontSize: 26, fontWeight: 600, fill: 'var(--color-ink)' }}
        >
          {Math.round(clamped * 100)}%
        </text>
      </svg>
      <p className="font-display text-base text-ink -mt-1">{label}</p>
      <p className="font-body text-xs font-medium" style={{ color: band.color }}>
        {band.label}
      </p>
    </div>
  )
}
