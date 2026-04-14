import { useState, useEffect } from 'react'
import { User, Zap, BookOpen, Plus, X, Save, Loader2 } from 'lucide-react'
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

      </div>
    </div>
  )
}
