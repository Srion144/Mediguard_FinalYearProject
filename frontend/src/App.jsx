import { useState } from 'react'
import PatientForm from './components/PatientForm'
import ResultsPanel from './components/ResultsPanel'
import { getPrediction } from './api/predictApi'

export default function App() {
  const [result, setResult] = useState(null)
  const [errorMessages, setErrorMessages] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(patient) {
    setLoading(true)
    setErrorMessages(null)
    setResult(null)
    try {
      const response = await getPrediction(patient)
      if (response.error) {
        setErrorMessages(response.messages)
      } else {
        setResult(response)
      }
    } catch (err) {
      setErrorMessages([`Could not reach the backend: ${err.message}`])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-canvas">
      <header className="max-w-5xl mx-auto px-6 pt-10 pb-6">
        <h1 className="font-display text-3xl text-ink">MediGuard AI</h1>
        <p className="font-body text-sm text-muted mt-1">
          Diabetes &amp; cardiovascular risk screening, built for primary-care use.
        </p>
      </header>

      <main className="max-w-5xl mx-auto px-6 pb-16 grid md:grid-cols-2 gap-6 items-start">
        <PatientForm onSubmit={handleSubmit} loading={loading} />
        <ResultsPanel result={result} errorMessages={errorMessages} />
      </main>
    </div>
  )
}
