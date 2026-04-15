import { useState, useEffect } from 'react'
import {
  BarChart2, Users, Flame, Zap, CheckSquare, TrendingUp, Award,
  Loader2, RefreshCw, FileText, X, AlertTriangle, Lightbulb,
  ChevronRight, Download, Shield, Target, Grid3x3,
} from 'lucide-react'
import { analytics } from '../api'

// ── Small reusable components ─────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = 'text-accent-600', bg = 'bg-accent-50' }) {
  return (
    <div className="card p-4 flex items-start gap-3">
      <div className={`w-9 h-9 rounded-xl ${bg} flex items-center justify-center shrink-0`}>
        <Icon size={16} className={color} />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900 leading-tight">{value}</p>
        <p className="text-xs font-medium text-gray-500">{label}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

function MiniBar({ value, max }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-6 bg-gray-100 rounded-sm overflow-hidden" style={{ height: 40 }}>
        <div className="w-6 bg-accent-400 rounded-sm transition-all duration-700"
          style={{ height: `${pct}%`, marginTop: `${100 - pct}%` }} />
      </div>
    </div>
  )
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

// ── Skill Gap row — clickable ─────────────────────────────────────────────────

function GapRow({ label, value, max, color, selected, onClick }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 rounded-lg px-2 py-1.5 -mx-2 transition-all group
        ${selected ? 'bg-accent-50 ring-1 ring-accent-200' : 'hover:bg-gray-50'}`}
    >
      <p className="text-xs text-gray-600 w-40 shrink-0 truncate text-left">{label}</p>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-2 rounded-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs font-semibold text-gray-700 w-8 text-right">{value}</p>
      <ChevronRight size={12} className={`shrink-0 transition-colors ${selected ? 'text-accent-500' : 'text-gray-300 group-hover:text-gray-400'}`} />
    </button>
  )
}

// ── Skill Gap Training Plan panel ─────────────────────────────────────────────

function GapPlanPanel({ skill, onClose }) {
  const [loading, setLoading] = useState(true)
  const [data, setData]       = useState(null)
  const [error, setError]     = useState('')

  useEffect(() => {
    analytics.gapPlan(skill)
      .then(d => { if (d.error) setError(d.error); else setData(d) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [skill])

  const downloadPdf = () => {
    if (!data) return
    const lines = [
      `Training Campaign: ${data.campaign_name || skill}`,
      `Skill: ${data.skill}  |  Employees affected: ${data.total_affected}`,
      `Estimated duration: ${data.estimated_weeks} weeks  |  Cost: ${data.estimated_cost || 'TBD'}`,
      '',
      `Approach: ${data.approach}`,
      '',
      ...(data.phases || []).flatMap(p => [
        `── ${p.name}  [${p.week_range}]`,
        `Focus: ${p.focus}`,
        ...(p.activities || []).map(a => `  • ${a}`),
        `Deliverable: ${p.deliverable}`,
        '',
      ]),
      'QUICK WINS THIS WEEK',
      ...(data.quick_wins || []).map(q => `  ✓ ${q}`),
      '',
      'SUCCESS METRICS',
      ...(data.success_metrics || []).map(m => `  • ${m}`),
      '',
      `ROI: ${data.roi_note || ''}`,
    ]
    const escaped = lines.join('\n').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    const win = window.open('', '_blank', 'width=820,height=700')
    if (!win) { alert('Allow popups to download as PDF.'); return }
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>${data.campaign_name || skill} — Training Plan</title>
<style>body{font-family:Georgia,serif;max-width:760px;margin:48px auto;font-size:13px;line-height:1.8;color:#111;padding:0 24px}
h1{font-size:18px;font-weight:bold;margin-bottom:4px}pre{white-space:pre-wrap;font-family:inherit}
@media print{body{margin:0}}</style>
</head><body><h1>${data.campaign_name || skill} — Training Campaign</h1>
<pre>${escaped}</pre><script>window.onload=()=>{window.print()}<\/script></body></html>`)
    win.document.close()
  }

  const PHASE_COLORS = ['bg-blue-600', 'bg-violet-600', 'bg-emerald-600']

  return (
    <div className="mt-4 card overflow-hidden border-accent-200 border">
      {/* Panel header */}
      <div className="flex items-center justify-between px-5 py-3 bg-accent-600 text-white">
        <div>
          <p className="text-xs font-semibold opacity-80">Training Campaign</p>
          <p className="text-sm font-bold">{skill}</p>
        </div>
        <button onClick={onClose} className="p-1 rounded hover:bg-white/20 transition-colors">
          <X size={15} />
        </button>
      </div>

      <div className="p-5">
        {loading && (
          <div className="flex items-center justify-center py-8 gap-2 text-sm text-gray-500">
            <Loader2 size={16} className="animate-spin text-accent-500" />
            Generating training campaign…
          </div>
        )}
        {error && <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{error}</p>}

        {data && (
          <div className="space-y-5">
            {/* Campaign overview */}
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <p className="text-sm font-semibold text-gray-900 mb-1">{data.campaign_name}</p>
                <p className="text-xs text-gray-500">{data.approach}</p>
              </div>
              <button
                onClick={downloadPdf}
                className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 border border-gray-200 hover:border-gray-400 px-2.5 py-1.5 rounded-lg transition-colors shrink-0"
              >
                <Download size={12} /> PDF
              </button>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-lg font-bold text-gray-900">{data.total_affected}</p>
                <p className="text-xs text-gray-500">employees affected</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-lg font-bold text-gray-900">{data.estimated_weeks}w</p>
                <p className="text-xs text-gray-500">to close gap</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs font-bold text-gray-900 leading-tight">{data.estimated_cost || 'Low'}</p>
                <p className="text-xs text-gray-500">estimated cost</p>
              </div>
            </div>

            {/* Affected employees */}
            {data.affected_employees?.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-500 mb-2">Employees to upskill</p>
                <div className="flex flex-wrap gap-2">
                  {data.affected_employees.map(name => (
                    <div key={name} className="flex items-center gap-1.5 bg-gray-50 border border-gray-200 rounded-full px-2.5 py-1">
                      <div className="w-5 h-5 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-[10px] font-bold shrink-0">
                        {name[0].toUpperCase()}
                      </div>
                      <span className="text-xs text-gray-700">{name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Training phases */}
            {data.phases?.length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Training Phases</p>
                {data.phases.map((phase, i) => (
                  <div key={i} className="rounded-xl overflow-hidden border border-gray-100">
                    <div className={`px-4 py-2.5 ${PHASE_COLORS[i % 3]} text-white`}>
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold">{phase.name}</p>
                        <p className="text-xs opacity-70">{phase.week_range}</p>
                      </div>
                      <p className="text-xs opacity-80 mt-0.5">{phase.focus}</p>
                    </div>
                    <div className="p-3 bg-gray-50 space-y-1">
                      {phase.activities?.map((a, j) => (
                        <p key={j} className="text-xs text-gray-600 flex items-start gap-1.5">
                          <span className="w-1 h-1 rounded-full bg-gray-400 shrink-0 mt-1.5" />{a}
                        </p>
                      ))}
                      {phase.deliverable && (
                        <p className="text-xs text-gray-500 mt-2 pt-2 border-t border-gray-200">
                          <span className="font-semibold">Deliverable:</span> {phase.deliverable}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Quick wins + ROI */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {data.quick_wins?.length > 0 && (
                <div className="bg-emerald-50 border border-emerald-100 rounded-xl p-4">
                  <p className="text-xs font-semibold text-emerald-700 mb-2 flex items-center gap-1.5">
                    <Zap size={11} /> Quick wins this week
                  </p>
                  {data.quick_wins.map((q, i) => (
                    <p key={i} className="text-xs text-emerald-800 flex items-start gap-1.5 mb-1">
                      <CheckSquare size={11} className="shrink-0 mt-0.5 text-emerald-600" />{q}
                    </p>
                  ))}
                </div>
              )}
              {data.roi_note && (
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                  <p className="text-xs font-semibold text-blue-700 mb-2 flex items-center gap-1.5">
                    <TrendingUp size={11} /> Business ROI
                  </p>
                  <p className="text-xs text-blue-800">{data.roi_note}</p>
                  {data.success_metrics?.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-blue-100 space-y-0.5">
                      {data.success_metrics.map((m, i) => (
                        <p key={i} className="text-xs text-blue-700 flex items-start gap-1">
                          <span className="shrink-0 mt-0.5">•</span> {m}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Executive Brief modal ─────────────────────────────────────────────────────

function BriefModal({ data, onClose }) {
  const GRADE_COLOR = {
    'A+': 'text-emerald-600', A: 'text-emerald-600', 'A-': 'text-emerald-500',
    'B+': 'text-blue-600',    B: 'text-blue-600',    'B-': 'text-blue-500',
    'C+': 'text-amber-500',   C: 'text-amber-500',
    D:    'text-red-500',
  }
  const SEV_COLOR = {
    critical: 'bg-red-50 border-red-200 text-red-700',
    high:     'bg-amber-50 border-amber-200 text-amber-700',
    medium:   'bg-blue-50 border-blue-200 text-blue-700',
  }
  const TIMELINE_COLOR = {
    'This week': 'bg-red-50 text-red-700',
    '30 days':   'bg-amber-50 text-amber-700',
    '60 days':   'bg-blue-50 text-blue-700',
    '90 days':   'bg-emerald-50 text-emerald-700',
  }
  const grade = data.org_health_grade || 'B'
  const gradeColor = GRADE_COLOR[grade] || 'text-gray-600'

  const downloadPdf = () => {
    const lines = [
      'WORKFORCE INTELLIGENCE BRIEF',
      '─────────────────────────────',
      `Org Health Score: ${data.org_health_score}/100  Grade: ${grade}`,
      '',
      `EXECUTIVE SUMMARY`,
      data.headline,
      '',
      data.strategic_summary,
      '',
      'RISK AREAS',
      ...(data.risk_areas || []).map(r => `  [${r.severity?.toUpperCase()}] ${r.skill} — ${r.employees_affected} employees\n    ${r.business_impact}`),
      '',
      'STRATEGIC OPPORTUNITIES',
      ...(data.opportunities || []).map(o => `  • ${o.title}: ${o.description}`),
      '',
      'PEOPLE TO WATCH',
      ...(data.top_performers || []).map(p => `  ★ ${p.name}: ${p.highlight}`),
      '',
      'PRIORITY ACTIONS',
      ...(data.priority_actions || []).map(a => `  [${a.timeline}] [${a.owner}] ${a.action}\n    → ${a.expected_outcome}`),
    ]
    const escaped = lines.join('\n').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    const win = window.open('', '_blank', 'width=820,height=700')
    if (!win) { alert('Allow popups to download as PDF.'); return }
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Workforce Intelligence Brief</title>
<style>body{font-family:Georgia,serif;max-width:760px;margin:48px auto;font-size:13px;line-height:1.8;color:#111;padding:0 24px}
h1{font-size:20px;font-weight:bold}pre{white-space:pre-wrap;font-family:inherit}
@media print{body{margin:0}}</style>
</head><body><h1>Workforce Intelligence Brief</h1>
<pre>${escaped}</pre><script>window.onload=()=>{window.print()}<\/script></body></html>`)
    win.document.close()
  }

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-accent-100 flex items-center justify-center">
              <FileText size={15} className="text-accent-600" />
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-900">Workforce Intelligence Brief</p>
              <p className="text-xs text-gray-400">AI-generated executive report — confidential</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={downloadPdf}
              className="flex items-center gap-1.5 text-xs border border-gray-200 hover:border-gray-400 text-gray-600 hover:text-gray-900 px-3 py-1.5 rounded-lg transition-colors"
            >
              <Download size={12} /> Download PDF
            </button>
            <button onClick={onClose} className="p-1.5 text-gray-400 hover:text-gray-700 rounded-lg hover:bg-gray-100 transition-colors">
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">

          {/* Score + headline */}
          <div className="flex items-center gap-6 p-5 bg-gray-50 rounded-2xl border border-gray-100">
            {/* Score ring */}
            <div className="shrink-0 text-center">
              {(() => {
                const score = data.org_health_score || 0
                const CIRC = 2 * Math.PI * 38
                const offset = CIRC * (1 - score / 100)
                const color = score >= 70 ? '#10b981' : score >= 50 ? '#3b82f6' : score >= 35 ? '#f59e0b' : '#ef4444'
                return (
                  <svg viewBox="0 0 100 100" width="88" height="88">
                    <circle cx="50" cy="50" r="38" fill="none" stroke="#f3f4f6" strokeWidth="8" />
                    <circle cx="50" cy="50" r="38" fill="none" stroke={color} strokeWidth="8"
                      strokeDasharray={CIRC} strokeDashoffset={offset}
                      strokeLinecap="round" transform="rotate(-90 50 50)"
                      style={{ transition: 'stroke-dashoffset 1.2s ease' }} />
                    <text x="50" y="46" textAnchor="middle" dominantBaseline="central"
                      fontSize="16" fontWeight="bold" fill={color}>{score}</text>
                    <text x="50" y="64" textAnchor="middle" fontSize="8" fill="#9ca3af">/ 100</text>
                  </svg>
                )
              })()}
              <p className={`text-2xl font-black mt-0 ${gradeColor}`}>{grade}</p>
              <p className="text-[10px] text-gray-400">Org Health</p>
            </div>

            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-gray-900 mb-2 leading-snug">{data.headline}</p>
              <p className="text-xs text-gray-500 leading-relaxed">{data.strategic_summary}</p>
              {data._analytics && (
                <div className="flex gap-4 mt-3">
                  {[
                    ['Engagement', `${Math.round((data._analytics.active_learners / Math.max(1, data._analytics.total_learners)) * 100)}%`],
                    ['Completion', `${data._analytics.completion_rate}%`],
                    ['Avg Streak', `${data._analytics.avg_streak}d`],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <p className="text-sm font-bold text-gray-800">{value}</p>
                      <p className="text-[10px] text-gray-400">{label}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Risk areas */}
          {data.risk_areas?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <AlertTriangle size={12} className="text-red-400" /> Risk Areas
              </p>
              <div className="space-y-2">
                {data.risk_areas.map((r, i) => (
                  <div key={i} className={`flex items-start gap-3 p-3 rounded-xl border ${SEV_COLOR[r.severity] || 'bg-gray-50 border-gray-100 text-gray-700'}`}>
                    <div className="shrink-0 mt-0.5">
                      <Shield size={13} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <p className="text-xs font-semibold">{r.skill}</p>
                        <span className="text-[10px] opacity-60 uppercase font-bold">{r.severity}</span>
                        <span className="text-[10px] opacity-60 ml-auto">{r.employees_affected} affected</span>
                      </div>
                      <p className="text-xs opacity-80">{r.business_impact}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Opportunities */}
          {data.opportunities?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Lightbulb size={12} className="text-amber-400" /> Strategic Opportunities
              </p>
              <div className="space-y-2">
                {data.opportunities.map((o, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-amber-50 border border-amber-100 rounded-xl">
                    <Lightbulb size={13} className="text-amber-500 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-semibold text-amber-800 mb-0.5">{o.title}</p>
                      <p className="text-xs text-amber-700">{o.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top performers */}
          {data.top_performers?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Award size={12} className="text-amber-500" /> People to Watch
              </p>
              <div className="space-y-2">
                {data.top_performers.map((p, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100">
                    <div className="w-8 h-8 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-sm font-semibold shrink-0">
                      {p.name?.[0]?.toUpperCase()}
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-gray-800">{p.name}</p>
                      <p className="text-xs text-gray-500">{p.highlight}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Priority actions */}
          {data.priority_actions?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Target size={12} className="text-accent-500" /> Priority Actions
              </p>
              <div className="space-y-2">
                {data.priority_actions.map((a, i) => (
                  <div key={i} className="p-3 border border-gray-100 rounded-xl hover:border-gray-200 transition-colors">
                    <div className="flex items-start gap-3">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 mt-0.5 ${TIMELINE_COLOR[a.timeline] || 'bg-gray-100 text-gray-600'}`}>
                        {a.timeline}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-gray-800">{a.action}</p>
                        <p className="text-xs text-gray-400 mt-0.5">
                          <span className="text-gray-500 font-medium">Owner: {a.owner}</span>
                          {a.expected_outcome && <> · {a.expected_outcome}</>}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}

// ── Skill Matrix view ─────────────────────────────────────────────────────────

function SkillMatrix() {
  const [loading, setLoading] = useState(true)
  const [data, setData]       = useState(null)
  const [error, setError]     = useState('')

  useEffect(() => {
    analytics.skillMatrix()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const downloadCsv = () => {
    if (!data) return
    const header = ['Employee', 'Current Role', ...data.skills]
    const rows = data.employees.map(emp => [
      emp.username,
      emp.current_role || '',
      ...emp.has.map(v => (v ? 'Yes' : '')),
    ])
    const csv = [header, ...rows]
      .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
      .join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = 'org_skill_matrix.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return (
    <div className="flex-1 flex items-center justify-center py-20">
      <Loader2 size={24} className="animate-spin text-accent-500" />
    </div>
  )
  if (error) return (
    <div className="m-6 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
  )
  if (!data) return null

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-sm font-semibold text-gray-800">Employee × Skill Matrix</p>
          <p className="text-xs text-gray-400">{data.employees.length} employees · top {data.skills.length} skills by frequency</p>
        </div>
        <button
          onClick={downloadCsv}
          className="flex items-center gap-1.5 text-xs border border-gray-200 hover:border-gray-400 text-gray-600 hover:text-gray-900 px-3 py-1.5 rounded-lg transition-colors"
        >
          <Download size={12} /> Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="card overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-100">
              <th className="text-left px-4 py-3 font-semibold text-gray-600 w-36 min-w-[9rem]">Employee</th>
              <th className="text-left px-4 py-3 font-semibold text-gray-600 w-40 min-w-[10rem]">Current Role</th>
              {data.skills.map(skill => (
                <th key={skill} className="px-3 py-3 font-semibold text-gray-600 text-center min-w-[6rem]">
                  <span className="block max-w-[5rem] mx-auto leading-tight">{skill}</span>
                </th>
              ))}
              <th className="px-3 py-3 font-semibold text-gray-500 text-center min-w-[4rem]">Score</th>
            </tr>
          </thead>
          <tbody>
            {data.employees.map((emp, idx) => {
              const score = emp.has.filter(Boolean).length
              const total = emp.has.length
              const pct   = total > 0 ? Math.round((score / total) * 100) : 0
              return (
                <tr key={emp.username} className={`border-b border-gray-50 hover:bg-gray-50 transition-colors ${idx % 2 === 0 ? '' : 'bg-gray-50/30'}`}>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-[10px] font-bold shrink-0">
                        {emp.username[0].toUpperCase()}
                      </div>
                      <span className="font-medium text-gray-800 truncate max-w-[7rem]">{emp.username}</span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-gray-500 truncate max-w-[10rem]">{emp.current_role || '—'}</td>
                  {emp.has.map((has, i) => (
                    <td key={i} className="px-3 py-2.5 text-center">
                      {has
                        ? <span className="inline-block w-5 h-5 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center text-[11px] font-bold">✓</span>
                        : <span className="inline-block w-5 h-5 rounded-full bg-gray-100 text-gray-300 flex items-center justify-center text-[11px]">—</span>
                      }
                    </td>
                  ))}
                  <td className="px-3 py-2.5 text-center">
                    <span className={`inline-block text-[10px] font-bold px-1.5 py-0.5 rounded-full
                      ${pct >= 70 ? 'bg-emerald-100 text-emerald-700' : pct >= 40 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-600'}`}>
                      {score}/{total}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Main AnalyticsPanel ───────────────────────────────────────────────────────

export default function AnalyticsPanel() {
  const [activeTab, setActiveTab] = useState('dashboard')

  const [data, setData]         = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState('')

  // Executive brief
  const [briefLoading, setBriefLoading] = useState(false)
  const [briefData, setBriefData]       = useState(null)
  const [briefError, setBriefError]     = useState('')

  // Skill gap drill-down
  const [selectedGap, setSelectedGap] = useState(null)

  const load = async () => {
    setLoading(true); setError('')
    try {
      setData(await analytics.dashboard())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const generateBrief = async () => {
    setBriefLoading(true); setBriefError(''); setBriefData(null)
    try {
      const d = await analytics.brief()
      if (d.error) { setBriefError(d.error); return }
      setBriefData(d)
    } catch (e) {
      setBriefError(e.message)
    } finally {
      setBriefLoading(false)
    }
  }

  const maxGap  = data ? Math.max(...data.skill_gaps.map(g => g.count), 1) : 1
  const maxAct  = data ? Math.max(...data.activity_week, 1) : 1
  const total   = data ? data.progress.saved + data.progress.in_progress + data.progress.done : 1

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 pt-5 pb-0 border-b border-gray-100 shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <BarChart2 size={20} className="text-accent-600" strokeWidth={2} />
              <h2 className="text-lg font-semibold text-gray-900">Org Analytics</h2>
            </div>
            <p className="text-sm text-gray-500">Workforce skill landscape — real-time overview</p>
          </div>
          <div className="flex items-center gap-2">
            {briefError && (
              <p className="text-xs text-red-500">{briefError}</p>
            )}
            {activeTab === 'dashboard' && (
              <>
                <button
                  onClick={generateBrief}
                  disabled={briefLoading || loading}
                  className="flex items-center gap-2 btn-primary text-xs py-2 px-4"
                >
                  {briefLoading
                    ? <><Loader2 size={13} className="animate-spin" />Generating…</>
                    : <><FileText size={13} />Executive Brief</>
                  }
                </button>
                <button
                  onClick={load}
                  disabled={loading}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  title="Refresh"
                >
                  <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1">
          {[
            { id: 'dashboard', label: 'Dashboard',    Icon: BarChart2  },
            { id: 'matrix',    label: 'Skill Matrix',  Icon: Grid3x3   },
          ].map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium border-b-2 transition-colors
                ${activeTab === id
                  ? 'border-accent-600 text-accent-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              <Icon size={13} />{label}
            </button>
          ))}
        </div>
      </div>

      {/* Skill Matrix tab */}
      {activeTab === 'matrix' && <SkillMatrix />}

      {/* Dashboard tab */}
      {activeTab === 'dashboard' && loading && (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 size={24} className="animate-spin text-accent-500" />
        </div>
      )}
      {activeTab === 'dashboard' && error && (
        <div className="m-6 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
      )}

      {activeTab === 'dashboard' && data && (
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">

          {/* Headline stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard icon={Users}       label="Total Learners"   value={data.total_learners}
              sub={`${data.active_learners} active this week`}     color="text-blue-600"    bg="bg-blue-50" />
            <StatCard icon={Flame}       label="Avg Streak"       value={`${data.avg_streak}d`}
              sub="days of continuous learning"                     color="text-orange-500"  bg="bg-orange-50" />
            <StatCard icon={Zap}         label="Skills Tracked"   value={data.total_skills}
              sub="across all employees"                            color="text-amber-500"   bg="bg-amber-50" />
            <StatCard icon={CheckSquare} label="Completion Rate"  value={`${data.completion_rate}%`}
              sub={`${data.progress.done} items completed`}         color="text-emerald-600" bg="bg-emerald-50" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

            {/* Skill Gap Heatmap — clickable */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp size={15} className="text-red-500" />
                <h3 className="text-sm font-semibold text-gray-800">Top Org Skill Gaps</h3>
                <span className="text-xs text-gray-400 ml-auto">employees lacking skill</span>
              </div>
              <p className="text-xs text-gray-400 mb-3">Click any gap to generate a training campaign</p>
              <div className="space-y-1">
                {data.skill_gaps.map((g, i) => (
                  <GapRow
                    key={g.skill}
                    label={g.skill}
                    value={g.count}
                    max={maxGap}
                    color={i < 3 ? 'bg-red-400' : i < 5 ? 'bg-amber-400' : 'bg-blue-300'}
                    selected={selectedGap === g.skill}
                    onClick={() => setSelectedGap(selectedGap === g.skill ? null : g.skill)}
                  />
                ))}
              </div>

              {/* Inline training campaign panel */}
              {selectedGap && (
                <GapPlanPanel
                  key={selectedGap}
                  skill={selectedGap}
                  onClose={() => setSelectedGap(null)}
                />
              )}
            </div>

            {/* Learning Activity */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 size={15} className="text-accent-500" />
                <h3 className="text-sm font-semibold text-gray-800">Learning Activity</h3>
                <span className="text-xs text-gray-400 ml-auto">last 7 days</span>
              </div>
              <div className="flex items-end gap-2 justify-around">
                {data.activity_week.map((v, i) => (
                  <div key={i} className="flex flex-col items-center gap-1.5">
                    <span className="text-xs font-semibold text-gray-600">{v}</span>
                    <MiniBar value={v} max={maxAct} />
                    <span className="text-[10px] text-gray-400">{DAYS[i]}</span>
                  </div>
                ))}
              </div>

              {/* Progress breakdown */}
              <div className="mt-5 pt-4 border-t border-gray-100">
                <p className="text-xs font-medium text-gray-500 mb-3">Progress Status</p>
                <div className="flex gap-2 mb-2">
                  <div className="h-2 rounded-full bg-gray-200 transition-all"
                    style={{ width: `${(data.progress.saved / total) * 100}%` }} />
                  <div className="h-2 rounded-full bg-amber-400 transition-all"
                    style={{ width: `${(data.progress.in_progress / total) * 100}%` }} />
                  <div className="h-2 rounded-full bg-emerald-400 transition-all"
                    style={{ width: `${(data.progress.done / total) * 100}%` }} />
                </div>
                <div className="flex gap-4 text-xs text-gray-500">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-200 inline-block" />Saved {data.progress.saved}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400 inline-block" />In Progress {data.progress.in_progress}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />Done {data.progress.done}</span>
                </div>
              </div>
            </div>

          </div>

          {/* Top Learners Leaderboard */}
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Award size={15} className="text-amber-500" />
              <h3 className="text-sm font-semibold text-gray-800">Top Learners</h3>
              <span className="text-xs text-gray-400 ml-auto">by learning streak</span>
            </div>
            <div className="space-y-2">
              {data.leaderboard.map((l, i) => (
                <div key={l.username} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0
                    ${i === 0 ? 'bg-amber-100 text-amber-700' : i === 1 ? 'bg-gray-100 text-gray-600' : i === 2 ? 'bg-orange-100 text-orange-700' : 'bg-gray-50 text-gray-400'}`}>
                    {i + 1}
                  </span>
                  <div className="w-8 h-8 rounded-full bg-accent-100 text-accent-700 flex items-center justify-center text-sm font-semibold shrink-0">
                    {l.username[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">{l.username}</p>
                    <p className="text-xs text-gray-400 truncate">{l.target_role || 'No target role set'}</p>
                  </div>
                  <div className="flex items-center gap-1 text-orange-500 shrink-0">
                    <Flame size={13} />
                    <span className="text-xs font-semibold">{l.streak}d</span>
                  </div>
                  <div className="text-xs text-gray-400 shrink-0">{l.skills?.length || 0} skills</div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}

      {/* Executive brief modal */}
      {briefData && <BriefModal data={briefData} onClose={() => setBriefData(null)} />}
    </div>
  )
}
