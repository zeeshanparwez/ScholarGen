import { useState, useEffect } from 'react'
import { BarChart2, Users, Flame, Zap, CheckSquare, TrendingUp, Award, Loader2, RefreshCw } from 'lucide-react'
import { analytics } from '../api'

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

function BarRow({ label, value, max, color = 'bg-accent-500' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div className="flex items-center gap-3">
      <p className="text-xs text-gray-600 w-40 shrink-0 truncate">{label}</p>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-2 rounded-full ${color} transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs font-semibold text-gray-700 w-8 text-right">{value}</p>
    </div>
  )
}

function MiniBar({ value, max }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-6 bg-gray-100 rounded-sm overflow-hidden" style={{ height: 40 }}>
        <div
          className="w-6 bg-accent-400 rounded-sm transition-all duration-700"
          style={{ height: `${pct}%`, marginTop: `${100 - pct}%` }}
        />
      </div>
    </div>
  )
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

export default function AnalyticsPanel() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')

  const load = async () => {
    setLoading(true); setError('')
    try {
      const d = await analytics.dashboard()
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const maxGap    = data ? Math.max(...data.skill_gaps.map(g => g.count), 1) : 1
  const maxAct    = data ? Math.max(...data.activity_week, 1) : 1
  const totalProg = data ? data.progress.saved + data.progress.in_progress + data.progress.done : 1

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <BarChart2 size={20} className="text-accent-600" strokeWidth={2} />
              <h2 className="text-lg font-semibold text-gray-900">Org Analytics</h2>
            </div>
            <p className="text-sm text-gray-500">Workforce skill landscape — real-time overview</p>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 size={24} className="animate-spin text-accent-500" />
        </div>
      )}

      {error && (
        <div className="m-6 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {data && (
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">

          {/* ── Headline stats ── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard
              icon={Users} label="Total Learners" value={data.total_learners}
              sub={`${data.active_learners} active this week`}
              color="text-blue-600" bg="bg-blue-50"
            />
            <StatCard
              icon={Flame} label="Avg Streak" value={`${data.avg_streak}d`}
              sub="days of continuous learning"
              color="text-orange-500" bg="bg-orange-50"
            />
            <StatCard
              icon={Zap} label="Skills Tracked" value={data.total_skills}
              sub="across all employees"
              color="text-amber-500" bg="bg-amber-50"
            />
            <StatCard
              icon={CheckSquare} label="Completion Rate" value={`${data.completion_rate}%`}
              sub={`${data.progress.done} items completed`}
              color="text-emerald-600" bg="bg-emerald-50"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

            {/* ── Skill Gap Heatmap ── */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp size={15} className="text-red-500" />
                <h3 className="text-sm font-semibold text-gray-800">Top Org Skill Gaps</h3>
                <span className="text-xs text-gray-400 ml-auto">employees lacking skill</span>
              </div>
              <div className="space-y-2.5">
                {data.skill_gaps.map((g, i) => (
                  <BarRow
                    key={g.skill}
                    label={g.skill}
                    value={g.count}
                    max={maxGap}
                    color={i < 3 ? 'bg-red-400' : i < 5 ? 'bg-amber-400' : 'bg-blue-300'}
                  />
                ))}
              </div>
            </div>

            {/* ── Learning Activity ── */}
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
                  <div
                    className="h-2 rounded-full bg-gray-200 transition-all"
                    style={{ width: `${(data.progress.saved / totalProg) * 100}%` }}
                  />
                  <div
                    className="h-2 rounded-full bg-amber-400 transition-all"
                    style={{ width: `${(data.progress.in_progress / totalProg) * 100}%` }}
                  />
                  <div
                    className="h-2 rounded-full bg-emerald-400 transition-all"
                    style={{ width: `${(data.progress.done / totalProg) * 100}%` }}
                  />
                </div>
                <div className="flex gap-4 text-xs text-gray-500">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-200 inline-block"/>Saved {data.progress.saved}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400 inline-block"/>In Progress {data.progress.in_progress}</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400 inline-block"/>Done {data.progress.done}</span>
                </div>
              </div>
            </div>

          </div>

          {/* ── Top Learners Leaderboard ── */}
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
    </div>
  )
}
