import { useState, useEffect } from 'react'
import {
  Map, ArrowRight, BookOpen, Zap, ExternalLink, Loader2,
  ChevronDown, ChevronUp, Users, Download, CheckCircle,
  Calendar, Target, Wrench,
} from 'lucide-react'
import { learningpath, profile as profileApi } from '../api'
import clsx from 'clsx'

const PHASE_COLORS = [
  'bg-blue-50 border-blue-200 text-blue-700',
  'bg-violet-50 border-violet-200 text-violet-700',
  'bg-emerald-50 border-emerald-200 text-emerald-700',
  'bg-amber-50 border-amber-200 text-amber-700',
]

const PERIOD_COLORS = ['bg-blue-600', 'bg-violet-600', 'bg-emerald-600']

const TABS = [
  { id: 'path',       label: 'Learning Path',    icon: Map   },
  { id: 'onboarding', label: '30-60-90 Onboarding', icon: Users },
]

function downloadAsPdf(content, title) {
  const escaped = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${title}</title>
<style>
  body{font-family:Georgia,serif;max-width:760px;margin:48px auto;line-height:1.8;font-size:13px;color:#111;padding:0 24px}
  h1{font-size:18px;font-weight:bold;margin-bottom:4px}
  h2{font-size:14px;font-weight:bold;margin:24px 0 6px;border-bottom:1px solid #ddd;padding-bottom:4px}
  h3{font-size:13px;font-weight:bold;margin:16px 0 4px}
  ul{padding-left:20px;margin:6px 0} li{margin:3px 0}
  .badge{display:inline-block;background:#f3f4f6;border-radius:4px;padding:2px 8px;margin:2px;font-size:11px}
  @media print{body{margin:0}}
</style></head><body>
<h1>${title}</h1>
<pre style="white-space:pre-wrap;font-family:inherit;font-size:13px">${escaped}</pre>
<script>window.onload=()=>{window.print()}<\/script>
</body></html>`
  const win = window.open('', '_blank', 'width=820,height=700')
  if (!win) { alert('Allow popups to download as PDF.'); return }
  win.document.write(html)
  win.document.close()
}

function pathToText(result) {
  const lines = [`Learning Path: ${result.current_role} → ${result.target_role}`, '']
  if (result.skill_gaps?.length) {
    lines.push('SKILL GAPS TO CLOSE', result.skill_gaps.map(s => `  • ${s}`).join('\n'), '')
  }
  result.phases?.forEach(p => {
    lines.push(`PHASE ${p.phase}: ${p.title}  [${p.duration}]`)
    lines.push(p.description)
    if (p.skills?.length) lines.push('  Skills: ' + p.skills.join(', '))
    lines.push('')
  })
  if (result.recommended_courses?.length) {
    lines.push('RECOMMENDED COURSES')
    result.recommended_courses.forEach(c => lines.push(`  • ${c.course_name}: ${c.url}`))
  }
  return lines.join('\n')
}

function onboardingToText(plan) {
  const lines = [`30-60-90 Day Onboarding Plan: ${plan.role}`, '', plan.overview, '']
  plan.periods?.forEach(p => {
    lines.push(`── ${p.label}: ${p.theme} ──`)
    lines.push('Goals:'); p.goals?.forEach(g => lines.push(`  • ${g}`))
    lines.push('Skills:'); p.skills_to_learn?.forEach(s => lines.push(`  • ${s}`))
    lines.push('Activities:'); p.key_activities?.forEach(a => lines.push(`  • ${a}`))
    lines.push(`Success: ${p.success_metric}`, '')
  })
  if (plan.key_tools?.length) lines.push('Key Tools: ' + plan.key_tools.join(', '))
  return lines.join('\n')
}

export default function LearningPathPanel() {
  const [tab, setTab]             = useState('path')

  // Learning path state
  const [currentRole, setCurrentRole] = useState('')
  const [targetRole, setTargetRole]   = useState('')
  const [profileLoaded, setProfileLoaded] = useState(false)
  const [jd, setJd]                   = useState('')
  const [showJd, setShowJd]           = useState(false)
  const [pathResult, setPathResult]   = useState(null)
  const [pathLoading, setPathLoading] = useState(false)
  const [pathError, setPathError]     = useState('')

  // Onboarding state
  const [hireRole, setHireRole]       = useState('')
  const [dept, setDept]               = useState('')
  const [obResult, setObResult]       = useState(null)
  const [obLoading, setObLoading]     = useState(false)
  const [obError, setObError]         = useState('')

  // Auto-fill from profile on mount
  useEffect(() => {
    profileApi.get().then(p => {
      if (p.current_role) setCurrentRole(p.current_role)
      if (p.target_role)  setTargetRole(p.target_role)
      setProfileLoaded(true)
    }).catch(() => setProfileLoaded(true))
  }, [])

  const generatePath = async (e) => {
    e.preventDefault()
    if (!currentRole.trim() || !targetRole.trim()) return
    setPathLoading(true); setPathError(''); setPathResult(null)
    try {
      const data = await learningpath.generate(currentRole.trim(), targetRole.trim(), jd.trim() || null)
      if (data.error) throw new Error(data.error)
      setPathResult(data)
    } catch (err) {
      setPathError(err.message)
    } finally {
      setPathLoading(false)
    }
  }

  const generateOnboarding = async (e) => {
    e.preventDefault()
    if (!hireRole.trim()) return
    setObLoading(true); setObError(''); setObResult(null)
    try {
      const data = await learningpath.onboarding(hireRole.trim(), dept.trim())
      if (data.error) throw new Error(data.error)
      setObResult(data)
    } catch (err) {
      setObError(err.message)
    } finally {
      setObLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100 shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <Map size={20} className="text-accent-600" />
          <h2 className="text-lg font-semibold text-gray-900">Learning Path</h2>
        </div>
        <p className="text-sm text-gray-500">Personalised roadmaps and new-hire onboarding plans</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-100 px-6 shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={clsx(
              'flex items-center gap-2 py-3 px-1 mr-6 text-sm font-medium border-b-2 -mb-px transition-colors',
              tab === id ? 'border-accent-600 text-accent-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            )}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">

          {/* ── LEARNING PATH TAB ── */}
          {tab === 'path' && (
            <>
              <form onSubmit={generatePath} className="card p-5 mb-6 space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs font-medium text-gray-700">Current Role / Skills</label>
                      {profileLoaded && currentRole && (
                        <span className="text-[10px] text-accent-600 bg-accent-50 px-1.5 py-0.5 rounded font-medium">from profile</span>
                      )}
                    </div>
                    <input
                      className="input-field text-sm"
                      placeholder="e.g. Backend Developer"
                      value={currentRole}
                      onChange={e => setCurrentRole(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs font-medium text-gray-700">Target Role / Goal</label>
                      {profileLoaded && targetRole && (
                        <span className="text-[10px] text-accent-600 bg-accent-50 px-1.5 py-0.5 rounded font-medium">from profile</span>
                      )}
                    </div>
                    <input
                      className="input-field text-sm"
                      placeholder="e.g. ML Engineer"
                      value={targetRole}
                      onChange={e => setTargetRole(e.target.value)}
                      required
                    />
                  </div>
                </div>

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
                  disabled={pathLoading}
                >
                  {pathLoading
                    ? <><Loader2 size={15} className="animate-spin" />Building your path…</>
                    : <><Map size={15} />Generate Learning Path</>
                  }
                </button>
              </form>

              {pathError && (
                <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl mb-4">
                  {pathError}
                </div>
              )}

              {pathResult && (
                <div className="space-y-5">
                  {/* Transition header + PDF */}
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-100">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-700 bg-white border border-gray-200 px-3 py-1.5 rounded-lg">
                        {pathResult.current_role}
                      </span>
                      <ArrowRight size={16} className="text-accent-500 shrink-0" />
                      <span className="text-sm font-semibold text-accent-700 bg-accent-50 border border-accent-200 px-3 py-1.5 rounded-lg">
                        {pathResult.target_role}
                      </span>
                    </div>
                    <button
                      onClick={() => downloadAsPdf(pathToText(pathResult), `Learning Path — ${pathResult.current_role} → ${pathResult.target_role}`)}
                      className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 border border-gray-200 hover:border-gray-400 px-2.5 py-1.5 rounded-lg transition-colors"
                    >
                      <Download size={13} /> PDF
                    </button>
                  </div>

                  {pathResult.skill_gaps?.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <Zap size={12} className="text-amber-500" /> Skill Gaps to Close
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {pathResult.skill_gaps.map(skill => (
                          <span key={skill} className="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-1 rounded-full font-medium">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {pathResult.phases?.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Your Roadmap</h3>
                      <div className="space-y-3">
                        {pathResult.phases.map((phase, idx) => (
                          <div key={phase.phase} className={`border rounded-xl p-4 ${PHASE_COLORS[idx % PHASE_COLORS.length]}`}>
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
                                <span key={skill} className="text-xs bg-white/60 px-2 py-0.5 rounded-md font-medium">{skill}</span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {pathResult.recommended_courses?.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                        <BookOpen size={12} className="text-accent-500" /> Recommended Courses
                      </h3>
                      <div className="space-y-2">
                        {pathResult.recommended_courses.map((course, idx) => (
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
                              <p className="text-sm font-medium text-gray-800 group-hover:text-accent-700 leading-tight">{course.course_name}</p>
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
            </>
          )}

          {/* ── ONBOARDING TAB ── */}
          {tab === 'onboarding' && (
            <>
              <form onSubmit={generateOnboarding} className="card p-5 mb-6 space-y-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1.5">New Hire Role</label>
                  <input
                    className="input-field text-sm"
                    placeholder="e.g. Backend Engineer, Data Analyst, Product Manager…"
                    value={hireRole}
                    onChange={e => setHireRole(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1.5">
                    Department <span className="text-gray-400 font-normal">(optional)</span>
                  </label>
                  <input
                    className="input-field text-sm"
                    placeholder="e.g. Platform Engineering, Growth, Data Science…"
                    value={dept}
                    onChange={e => setDept(e.target.value)}
                  />
                </div>
                <button
                  type="submit"
                  className="btn-primary w-full py-2.5 flex items-center justify-center gap-2"
                  disabled={obLoading}
                >
                  {obLoading
                    ? <><Loader2 size={15} className="animate-spin" />Generating plan…</>
                    : <><Calendar size={15} />Generate 30-60-90 Plan</>
                  }
                </button>
              </form>

              {obError && (
                <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl mb-4">
                  {obError}
                </div>
              )}

              {obResult && (
                <div className="space-y-5">
                  {/* Header + PDF */}
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-100">
                    <div>
                      <p className="text-xs text-gray-400 mb-0.5">30-60-90 Day Onboarding Plan</p>
                      <p className="text-sm font-semibold text-gray-900">{obResult.role}</p>
                      {obResult.overview && (
                        <p className="text-xs text-gray-500 mt-1 max-w-lg">{obResult.overview}</p>
                      )}
                    </div>
                    <button
                      onClick={() => downloadAsPdf(onboardingToText(obResult), `30-60-90 Onboarding — ${obResult.role}`)}
                      className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 border border-gray-200 hover:border-gray-400 px-2.5 py-1.5 rounded-lg transition-colors shrink-0 ml-4"
                    >
                      <Download size={13} /> PDF
                    </button>
                  </div>

                  {/* Periods */}
                  {obResult.periods?.map((period, idx) => (
                    <div key={period.label} className="card overflow-hidden">
                      <div className={`px-5 py-3 ${PERIOD_COLORS[idx]} text-white`}>
                        <p className="text-xs font-semibold opacity-80">{period.label}</p>
                        <p className="text-sm font-bold">{period.theme}</p>
                      </div>
                      <div className="p-5 grid grid-cols-1 md:grid-cols-3 gap-4">
                        {period.goals?.length > 0 && (
                          <div>
                            <p className="text-xs font-semibold text-gray-500 mb-2 flex items-center gap-1">
                              <Target size={11} /> Goals
                            </p>
                            <ul className="space-y-1">
                              {period.goals.map((g, i) => (
                                <li key={i} className="flex items-start gap-1.5 text-xs text-gray-700">
                                  <CheckCircle size={11} className="text-emerald-500 shrink-0 mt-0.5" /> {g}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {period.skills_to_learn?.length > 0 && (
                          <div>
                            <p className="text-xs font-semibold text-gray-500 mb-2 flex items-center gap-1">
                              <Zap size={11} className="text-amber-500" /> Skills to Learn
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {period.skills_to_learn.map(s => (
                                <span key={s} className="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded-full">{s}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {period.key_activities?.length > 0 && (
                          <div>
                            <p className="text-xs font-semibold text-gray-500 mb-2 flex items-center gap-1">
                              <Wrench size={11} /> Activities
                            </p>
                            <ul className="space-y-1">
                              {period.key_activities.map((a, i) => (
                                <li key={i} className="text-xs text-gray-700 flex items-start gap-1.5">
                                  <span className="w-1 h-1 rounded-full bg-gray-400 shrink-0 mt-1.5" /> {a}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                      {period.success_metric && (
                        <div className="px-5 pb-4">
                          <p className="text-xs text-gray-500 bg-gray-50 border border-gray-100 rounded-lg px-3 py-2">
                            <span className="font-semibold">Success at {period.label}:</span> {period.success_metric}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Key tools */}
                  {obResult.key_tools?.length > 0 && (
                    <div className="card p-4">
                      <p className="text-xs font-semibold text-gray-500 mb-3 flex items-center gap-1.5">
                        <Wrench size={12} /> Key Tools & Platforms
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {obResult.key_tools.map(t => (
                          <span key={t} className="text-xs bg-gray-50 text-gray-700 border border-gray-200 px-2.5 py-1 rounded-full">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

        </div>
      </div>
    </div>
  )
}
