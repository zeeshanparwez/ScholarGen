import { useState } from 'react'
import { Zap, ArrowRight, Loader2 } from 'lucide-react'
import { profile as profileApi } from '../api'

export default function OnboardingModal({ onDone }) {
  const [currentRole, setCurrentRole] = useState('')
  const [targetRole, setTargetRole] = useState('')
  const [saving, setSaving] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!currentRole.trim() || !targetRole.trim()) return
    setSaving(true)
    try {
      await profileApi.update({
        interests: [],
        skills: [],
        current_role: currentRole.trim(),
        target_role: targetRole.trim(),
      })
    } catch { /* non-critical — proceed anyway */ }
    localStorage.setItem('upskill_onboarded', '1')
    setSaving(false)
    onDone()
  }

  const skip = () => {
    localStorage.setItem('upskill_onboarded', '1')
    onDone()
  }

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8">
        {/* Logo */}
        <div className="flex items-center gap-2 mb-6">
          <div className="w-9 h-9 bg-accent-600 rounded-xl flex items-center justify-center">
            <Zap size={18} className="text-white" strokeWidth={2} />
          </div>
          <span className="font-semibold text-gray-900 text-lg tracking-tight">UpskillOS</span>
        </div>

        <h2 className="text-2xl font-bold text-gray-900 mb-1">Welcome aboard!</h2>
        <p className="text-gray-500 text-sm mb-7">
          Tell us where you are and where you're headed — we'll personalise everything for you.
        </p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1.5">
              What's your current role?
            </label>
            <input
              className="input-field"
              placeholder="e.g. Backend Developer, Data Analyst, Student"
              value={currentRole}
              onChange={e => setCurrentRole(e.target.value)}
              required
              autoFocus
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1.5">
              Where do you want to be?
            </label>
            <input
              className="input-field"
              placeholder="e.g. ML Engineer, Product Manager, Cloud Architect"
              value={targetRole}
              onChange={e => setTargetRole(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            disabled={saving}
            className="btn-primary w-full py-2.5 flex items-center justify-center gap-2 mt-2"
          >
            {saving ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <>
                Get started <ArrowRight size={15} />
              </>
            )}
          </button>
        </form>

        <button onClick={skip} className="w-full text-center text-xs text-gray-400 hover:text-gray-600 mt-4 transition-colors">
          Skip for now
        </button>
      </div>
    </div>
  )
}
