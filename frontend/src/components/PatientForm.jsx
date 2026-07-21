import { useState } from 'react'

const initialForm = {
  age: '',
  bmi: '',
  bp_diastolic: '',
  glucose_mgdl: '',
  cholesterol_cat: '1',
}

export default function PatientForm({ onSubmit, loading }) {
  const [form, setForm] = useState(initialForm)

  function handleChange(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onSubmit({
      age: Number(form.age),
      bmi: Number(form.bmi),
      bp_diastolic: Number(form.bp_diastolic),
      glucose_mgdl: Number(form.glucose_mgdl),
      cholesterol_cat: Number(form.cholesterol_cat),
    })
  }

  return (
    <form onSubmit={handleSubmit} className="bg-surface border border-border rounded-2xl p-6 space-y-4 h-fit">
      <h2 className="font-display text-xl text-ink">Patient details</h2>

      <Field
        label="Age (years)"
        value={form.age}
        onChange={(v) => handleChange('age', v)}
        placeholder="e.g. 52"
      />
      <Field
        label="BMI (kg/m²)"
        value={form.bmi}
        onChange={(v) => handleChange('bmi', v)}
        placeholder="e.g. 27.5"
        step="0.1"
      />
      <Field
        label="Diastolic blood pressure (mmHg)"
        value={form.bp_diastolic}
        onChange={(v) => handleChange('bp_diastolic', v)}
        placeholder="e.g. 88"
      />
      <Field
        label="Fasting glucose (mg/dL)"
        value={form.glucose_mgdl}
        onChange={(v) => handleChange('glucose_mgdl', v)}
        placeholder="e.g. 110"
      />

      <div>
        <label className="block text-sm font-body text-muted mb-1">Cholesterol</label>
        <select
          value={form.cholesterol_cat}
          onChange={(e) => handleChange('cholesterol_cat', e.target.value)}
          className="w-full border border-border rounded-lg px-3 py-2 font-body text-ink bg-surface focus:outline-none focus:ring-2 focus:ring-guard"
        >
          <option value="1">Normal</option>
          <option value="2">Above normal</option>
          <option value="3">Well above normal</option>
        </select>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-guard hover:bg-guard-deep text-white font-body font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? 'Assessing…' : 'Assess risk'}
      </button>
    </form>
  )
}

function Field({ label, value, onChange, placeholder, step }) {
  return (
    <div>
      <label className="block text-sm font-body text-muted mb-1">{label}</label>
      <input
        type="number"
        step={step || '1'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required
        className="w-full border border-border rounded-lg px-3 py-2 font-mono text-ink bg-surface focus:outline-none focus:ring-2 focus:ring-guard"
      />
    </div>
  )
}
