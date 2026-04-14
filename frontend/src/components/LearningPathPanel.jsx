import { useState } from 'react'
import { Map, ArrowRight, BookOpen, Zap, ExternalLink, Loader2, ChevronDown, ChevronUp } from 'lucide-react'
import { learningpath } from '../api'

const PHASE_COLORS = [
  'bg-blue-50 border-blue-200 text-blue-700',
  'bg-violet-50 border-violet-200 text-violet-700',
  'bg-emerald-50 border-emerald-200 text-emerald-700',
  'bg-amber-50 border-amber-200 text-amber-700',
]

export default function LearningPathPanel() {
  const [currentRole, setCurrentRole] = useState('')
  const [targetRole, setTargetRole] = useState('')
  const [jd, setJd] = useState('')
  const [showJd, setShowJd] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const generate = async (e) => {
    e.preventDefault()
    if (!currentRole.trim() || !targetRole.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await learningpath.generate(
        currentRole.trim(),
        targetRole.trim(),
        jd.trim() || null,
      )
      if (data.error) throw new Error(data.error)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-1">
            <Map size={20} className="text-accent-600" />
            <h2 className="text-lg font-semibold text-gray-900">Learning Path Generator</h2>
          </div>
          <p className="text-sm text-gray-500">
            Tell us where you are and where you want to go — we'll build a personalised roadmap with real courses.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={generate} className="card p-5 mb-6 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Current Role / Skills</label>
              <input
                className="input-field text-sm"
                placeholder="e.g. Backend Developer"
                value={currentRole}
                onChange={e => setCurrentRole(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Target Role / Goal</label>
              <input
                className="input-field text-sm"
                placeholder="e.g. ML Engineer"
                value={targetRole}
                onChange={e => setTargetRole(e.target.value)}
                required
              />
            </div>
          </div>

          {/* Optional JD */}
          <div>
            <button
              type="button"
              onClick={() => setShowJd(!showJd)}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-accent-600 transition-colors"
            >
              {showJd ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              Paste a job description for more precise results (optional)
            </button>
            {showJd && (
              <textarea
                className="input-field text-sm mt-2 resize-none"
                rows={5}
                placeholder="Paste the job description here…"
                value={jd}
                onChange={e => setJd(e.target.value)}
              />
            )}
          </div>

          <button
            type="submit"
            className="btn-primary w-full py-2.5 flex items-center justify-center gap-2"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 size={15} className="animate-spin" />
                Building your path…
              </>
            ) : (
              <>
                <Map size={15} />
                Generate Learning Path
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl mb-4">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-5">
            {/* Transition header */}
            <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-xl border border-gray-100">
              <span className="text-sm font-medium text-gray-700 bg-white border border-gray-200 px-3 py-1.5 rounded-lg">
                {result.current_role}
              </span>
              <ArrowRight size={16} className="text-accent-500 shrink-0" />
              <span className="text-sm font-semibold text-accent-700 bg-accent-50 border border-accent-200 px-3 py-1.5 rounded-lg">
                {result.target_role}
              </span>
            </div>

            {/* Skill gaps */}
            {result.skill_gaps?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Zap size={12} className="text-amber-500" />
                  Skill Gaps to Close
                </h3>
                <div className="flex flex-wrap gap-2">
                  {result.skill_gaps.map(skill => (
                    <span
                      key={skill}
                      className="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-1 rounded-full font-medium"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Phases */}
            {result.phases?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                  Your Roadmap
                </h3>
                <div className="space-y-3">
                  {result.phases.map((phase, idx) => (
                    <div
                      key={phase.phase}
                      className={`border rounded-xl p-4 ${PHASE_COLORS[idx % PHASE_COLORS.length]}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <span className="text-xs font-semibold opacity-70">Phase {phase.phase}</span>
                          <h4 className="font-semibold text-sm">{phase.title}</h4>
                        </div>
                        <span className="text-xs opacity-60 shrink-0 ml-3">{phase.duration}</span>
                      </div>
                      <p className="text-xs opacity-80 mb-2">{phase.description}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {phase.skills?.map(skill => (
                          <span key={skill} className="text-xs bg-white/60 px-2 py-0.5 rounded-md font-medium">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended courses */}
            {result.recommended_courses?.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <BookOpen size={12} className="text-accent-500" />
                  Recommended Courses
                </h3>
                <div className="space-y-2">
                  {result.recommended_courses.map((course, idx) => (
                    <a
                      key={idx}
                      href={course.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-3 p-3 rounded-xl border border-gray-100 hover:border-accent-200 hover:bg-accent-50 transition-all group"
                    >
                      <div className="w-7 h-7 rounded-lg bg-accent-100 flex items-center justify-center shrink-0 mt-0.5">
                        <BookOpen size={13} className="text-accent-600" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-gray-800 group-hover:text-accent-700 leading-tight">
                          {course.course_name}
                        </p>
                        <p className="text-xs text-gray-400 mt-0.5 line-clamp-1">{course.description}</p>
                      </div>
                      <ExternalLink size={13} className="text-gray-300 group-hover:text-accent-400 shrink-0 mt-1" />
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
