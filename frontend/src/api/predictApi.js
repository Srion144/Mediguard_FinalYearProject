import { mockPrediction } from '../mocks/mockPrediction'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

// Week 2: the backend didn't exist yet, so VITE_USE_MOCK=true made this
// return a hardcoded response shaped exactly like the real API. Week 3:
// with the real /predict endpoint running, VITE_USE_MOCK=false (the
// default in .env) calls it directly instead -- see getPrediction() below.
// No component that calls getPrediction() needs to know or care which one
// it's actually getting.
const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

async function getPredictionFromApi(patient) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  })
  if (!res.ok) {
    throw new Error(`Backend returned HTTP ${res.status}`)
  }
  return res.json()
}

function getPredictionMock(patient) {
  // Small artificial delay so loading states are visible even when
  // testing fully offline against the mock.
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockPrediction(patient)), 500)
  })
}

export function getPrediction(patient) {
  return USE_MOCK ? getPredictionMock(patient) : getPredictionFromApi(patient)
}
