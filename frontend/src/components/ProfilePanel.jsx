import { useState, useEffect } from 'react'
import { User, Zap, BookOpen, Plus, X, Save, Loader2, Copy, Check, Sparkles, Download, Target, TrendingUp, AlertCircle, ChevronRight } from 'lucide-react'
import { profile as profileApi } from '../api'

function TagList({ tags, onRemove, color = 'accent' }) {
  const colors = {
    accent: 'bg-accent-50 text-accent-700 border-accent-200',
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  }
  return (
    <div className="flex flex-wrap gap-2">
      {tags.map(tag => (
        <span
          key={tag}
          className={`flex items-center gap-1 text-xs border px-2.5 py-1 rounded-full font-medium ${colors[color]}`}
        >
          {tag}
          {onRemove && (
            <button onClick={() => onRemove(tag)} className="hover:opacity-60 ml-0.5">
              <X size={10} />
            </button>
          )}
        </span>
      ))}
    </div>
  )
}

function TagInput({ onAdd, placeholder }) {
  const [val, setVal] = useState('')
  const submit = () => {
    const trimmed = val.trim()
    if (trimmed) { onAdd(trimmed); setVal('') }
  }
  return (
    <div className="flex gap-2 mt-2">
      <input
        className="input-field text-sm flex-1"
        placeholder={placeholder}
        value={val}
        onChange={e => setVal(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && (e.preventDefault(), submit())}
      />
      <button type="button" onClick={submit} className="btn-primary px-3 py-1.5 text-xs flex items-center gap-1">
        <Plus size={13} /> Add
      </button>
    </div>
  )
}

export default function ProfilePanel() {
  const username = localStorage.getItem('sg_user') || 'User'
  const [data, setData] = useState(null)
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  // Editable copies
  const [interests, setInterests] = useState([])
  const [skills, setSkills] = useState([])
  const [currentRole, setCurrentRole] = useState('')
  const [targetRole, setTargetRole] = useState('')

  // Bio generator
  const [bioLoading, setBioLoading] = useState(false)
  const [bioResult, setBioResult]   = useState(null)
  const [bioError, setBioError]     = useState('')
  const [copiedField, setCopiedField] = useState(null)

  // Role Readiness
  const [readLoading, setReadLoading] = useState(false)
  const [readResult, setReadResult]   = useState(null)
  const [readError, setReadError]     = useState('')

  useEffect(() => {
    profileApi.get().then(d => {
      setData(d)
      setInterests(d.interests || [])
      setSkills(d.skills || [])
      setCurrentRole(d.current_role || '')
      setTargetRole(d.target_role || '')
    }).catch(() => {})
  }, [])

  const save = async () => {
    setSaving(true)
    try {
      await profileApi.update({ interests, skills, current_role: currentRole, target_role: targetRole })
      setData(prev => ({ ...prev, interests, skills, current_role: currentRole, target_role: targetRole }))
      setEditing(false)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const generateBio = async () => {
    setBioLoading(true); setBioError(''); setBioResult(null)
    try {
      const data = await profileApi.generateBio()
      if (data.error) { setBioError(data.error); return }
      setBioResult(data)
    } catch (e) {
      setBioError(e.message)
    } finally {
      setBioLoading(false)
    }
  }

  const checkReadiness = async () => {
    setReadLoading(true); setReadError(''); setReadResult(null)
    try {
      const d = await profileApi.readiness()
      if (d.error) { setReadError(d.error); return }
      setReadResult(d)
    } catch (e) {
      setReadError(e.message)
    } finally {
      setReadLoading(false)
    }
  }

  const copyField = async (text, field) => {
    await navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  const cancel = () => {
    setInterests(data?.interests || [])
    setSkills(data?.skills || [])
    setCurrentRole(data?.current_role || '')
    setTargetRole(data?.target_role || '')
    setEditing(false)
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-xl mx-auto space-y-5">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <User size={20} className="text-accent-600" />
            <h2 className="text-lg font-semibold text-gray-900">Skill Profile</h2>
          </div>
          {!editing ? (
            <button onClick={() => setEditing(true)} className="btn-ghost text-xs py-1.5 px-3">
              Edit profile
            </button>
          ) : (
            <div className="flex gap-2">
              <button onClick={cancel} className="btn-ghost text-xs py-1.5 px-3 text-gray-500">Cancel</button>
              <button onClick={save} disabled={saving} className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5">
                {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
                Save
              </button>
            </div>
          )}
        </div>

        {saved && (
          <div className="text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-3 py-2 rounded-lg">
            Profile saved successfully.
          </div>
        )}

        {/* Identity card */}
        <div className="card p-5">
          <div className="flex items-center gap-4 mb-5">
            <div className="w-14 h-14 rounded-2xl bg-accent-100 text-accent-700 flex items-center justify-center text-2xl font-bold">
              {username[0]?.toUpperCase()}
            </div>
            <div>
              <p className="font-semibold text-gray-900">{username}</p>
              <p className="text-xs text-gray-400 mt-0.5">
                {data?.last_updated
                  ? `Profile updated ${new Date(data.last_updated).toLocaleDateString()}`
                  : 'Profile not yet updated'}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1.5">Current Role</label>
              {editing ? (
                <input
                  className="input-field text-sm"
                  placeholder="e.g. Backend Developer"
                  value={currentRole}
                  onChange={e => setCurrentRole(e.target.value)}
                />
              ) : (
                <p className="text-sm text-gray-800 font-medium">
                  {currentRole || <span className="text-gray-400 italic">Not set</span>}
                </p>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1.5">Target Role</label>
              {editing ? (
                <input
                  className="input-field text-sm"
                  placeholder="e.g. ML Engineer"
                  value={targetRole}
                  onChange={e => setTargetRole(e.target.value)}
                />
              ) : (
                <p className="text-sm text-gray-800 font-medium">
                  {targetRole || <span className="text-gray-400 italic">Not set</span>}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Skills */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Zap size={15} className="text-amber-500" />
            <h3 className="text-sm font-semibold text-gray-800">Skills</h3>
            <span className="text-xs text-gray-400 ml-auto">Auto-built from your conversations</span>
          </div>
          {skills.length > 0 ? (
            <TagList
              tags={skills}
              onRemove={editing ? (t) => setSkills(skills.filter(s => s !== t)) : null}
              color="emerald"
            />
          ) : (
            <p className="text-xs text-gray-400 italic">No skills detected yet — start chatting!</p>
          )}
          {editing && <TagInput onAdd={t => !skills.includes(t) && setSkills([...skills, t])} placeholder="Add a skill…" />}
        </div>

        {/* Interests */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen size={15} className="text-accent-500" />
            <h3 className="text-sm font-semibold text-gray-800">Interests</h3>
            <span className="text-xs text-gray-400 ml-auto">Auto-built from your conversations</span>
          </div>
          {interests.length > 0 ? (
            <TagList
              tags={interests}
              onRemove={editing ? (t) => setInterests(interests.filter(i => i !== t)) : null}
              color="accent"
            />
          ) : (
            <p className="text-xs text-gray-400 italic">No interests detected yet — start chatting!</p>
          )}
          {editing && <TagInput onAdd={t => !interests.includes(t) && setInterests([...interests, t])} placeholder="Add an interest…" />}
        </div>

        {/* Role Readiness Score */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Target size={15} className="text-accent-500" />
              <h3 className="text-sm font-semibold text-gray-800">Role Readiness Score</h3>
            </div>
            <button
              onClick={checkReadiness}
              disabled={readLoading}
              className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              {readLoading
                ? <><Loader2 size={12} className="animate-spin" /> Analyzing…</>
                : <><TrendingUp size={12} /> Check Readiness</>
              }
            </button>
          </div>
          <p className="text-xs text-gray-400 mb-3">
            AI-scored readiness for your target role based on your current skills.
          </p>

          {readError && (
            <p className="text-xs text-red-600 bg-red-50 border border-red-200 px-3 py-2 rounded-lg flex items-center gap-2">
              <AlertCircle size={12} /> {readError}
            </p>
          )}

          {readResult && (() => {
            const CIRC = 2 * Math.PI * 38
            const offset = CIRC * (1 - readResult.score / 100)
            const color = readResult.score >= 70 ? '#10b981' : readResult.score >= 40 ? '#f59e0b' : '#ef4444'
            const label = readResult.score >= 70 ? 'Job Ready' : readResult.score >= 40 ? 'Progressing' : 'Early Stage'
            return (
              <div className="space-y-4 mt-1">
                {/* Score ring + summary */}
                <div className="flex items-center gap-5">
                  <svg viewBox="0 0 100 100" width="96" height="96" className="shrink-0">
                    <circle cx="50" cy="50" r="38" fill="none" stroke="#f3f4f6" strokeWidth="8" />
                    <circle
                      cx="50" cy="50" r="38" fill="none"
                      stroke={color} strokeWidth="8"
                      strokeDasharray={CIRC}
                      strokeDashoffset={offset}
                      strokeLinecap="round"
                      transform="rotate(-90 50 50)"
                      style={{ transition: 'stroke-dashoffset 1s ease' }}
                    />
                    <text x="50" y="46" textAnchor="middle" dominantBaseline="central"
                      fontSize="18" fontWeight="bold" fill={color}>{readResult.score}%</text>
                    <text x="50" y="64" textAnchor="middle" fontSize="9" fill="#9ca3af">{label}</text>
                  </svg>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-gray-700 mb-1">
                      {readResult.target_role || 'Target Role'}
                    </p>
                    <p className="text-xs text-gray-500 leading-relaxed">{readResult.summary}</p>
                  </div>
                </div>

                {/* Matched skills */}
                {readResult.matched_skills?.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-gray-500 mb-1.5">You already have</p>
                    <div className="flex flex-wrap gap-1.5">
                      {readResult.matched_skills.map(s => (
                        <span key={s} className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-200 px-2.5 py-0.5 rounded-full font-medium">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Missing skills */}
                {readResult.missing_skills?.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-gray-500 mb-1.5">Skills to build</p>
                    <div className="flex flex-wrap gap-1.5">
                      {readResult.missing_skills.map(s => (
                        <span key={s} className="text-xs bg-red-50 text-red-600 border border-red-200 px-2.5 py-0.5 rounded-full font-medium">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Next actions */}
                {readResult.next_actions?.length > 0 && (
                  <div className="bg-accent-50 rounded-lg p-3 border border-accent-100">
                    <p className="text-xs font-semibold text-accent-700 mb-2">Your next steps</p>
                    <div className="space-y-1.5">
                      {readResult.next_actions.map((a, i) => (
                        <div key={i} className="flex items-start gap-2">
                          <ChevronRight size={11} className="text-accent-500 mt-0.5 shrink-0" />
                          <p className="text-xs text-accent-800">{a}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )
          })()}
        </div>

        {/* LinkedIn Bio Generator */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Sparkles size={15} className="text-accent-500" />
              <h3 className="text-sm font-semibold text-gray-800">LinkedIn Bio Generator</h3>
            </div>
            <button
              onClick={generateBio}
              disabled={bioLoading}
              className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              {bioLoading
                ? <><Loader2 size={12} className="animate-spin" /> Generating…</>
                : <><Sparkles size={12} /> Generate</>
              }
            </button>
          </div>
          <p className="text-xs text-gray-400 mb-3">
            AI-crafted headline, bio, and elevator pitch based on your skill profile.
          </p>

          {bioError && (
            <p className="text-xs text-red-600 bg-red-50 border border-red-200 px-3 py-2 rounded-lg">{bioError}</p>
          )}

          {bioResult && (
            <div className="space-y-3 mt-1">
              {bioResult.headline && (
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-gray-500">Headline</p>
                    <button
                      onClick={() => copyField(bioResult.headline, 'headline')}
                      className="text-xs text-gray-400 hover:text-gray-700 flex items-center gap-1"
                    >
                      {copiedField === 'headline' ? <Check size={11} className="text-green-600" /> : <Copy size={11} />}
                      {copiedField === 'headline' ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                  <p className="text-sm text-gray-800 font-medium">{bioResult.headline}</p>
                </div>
              )}
              {bioResult.bio && (
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-gray-500">About / Bio</p>
                    <button
                      onClick={() => copyField(bioResult.bio, 'bio')}
                      className="text-xs text-gray-400 hover:text-gray-700 flex items-center gap-1"
                    >
                      {copiedField === 'bio' ? <Check size={11} className="text-green-600" /> : <Copy size={11} />}
                      {copiedField === 'bio' ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{bioResult.bio}</p>
                </div>
              )}
              {bioResult.elevator_pitch && (
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-gray-500">Elevator Pitch (30 sec)</p>
                    <button
                      onClick={() => copyField(bioResult.elevator_pitch, 'pitch')}
                      className="text-xs text-gray-400 hover:text-gray-700 flex items-center gap-1"
                    >
                      {copiedField === 'pitch' ? <Check size={11} className="text-green-600" /> : <Copy size={11} />}
                      {copiedField === 'pitch' ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{bioResult.elevator_pitch}</p>
                </div>
              )}

              <button
                onClick={() => {
                  const content = [
                    bioResult.headline && `HEADLINE\n${bioResult.headline}`,
                    bioResult.bio && `\nABOUT / BIO\n${bioResult.bio}`,
                    bioResult.elevator_pitch && `\nELEVATOR PITCH\n${bioResult.elevator_pitch}`,
                  ].filter(Boolean).join('\n')
                  const escaped = content.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  const win = window.open('', '_blank', 'width=800,height=600')
                  if (!win) { alert('Allow popups to download as PDF.'); return }
                  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>LinkedIn Bio</title>
                  <style>body{font-family:Georgia,serif;max-width:700px;margin:48px auto;line-height:1.8;font-size:13px;color:#111;padding:0 24px}
                  h2{font-size:15px;font-weight:bold;margin:24px 0 8px;border-bottom:1px solid #ddd;padding-bottom:6px}pre{white-space:pre-wrap;font-family:inherit}</style>
                  </head><body><pre>${escaped}</pre><script>window.onload=()=>{window.print()}<\/script></body></html>`)
                  win.document.close()
                }}
                className="flex items-center gap-1.5 text-xs text-accent-600 hover:text-accent-700 font-medium border border-accent-200 hover:border-accent-400 bg-accent-50 px-3 py-1.5 rounded-lg transition-colors w-full justify-center"
              >
                <Download size={13} /> Download as PDF
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
