import { useState, useRef } from 'react'
import {
  Briefcase, FileText, Loader2, CheckCircle, XCircle, Zap,
  ChevronRight, AlertCircle, Mail, Youtube, Plus, Trash2,
  Copy, Check, FileUp, Download,
} from 'lucide-react'
import clsx from 'clsx'
import { career } from '../api'

const TABS = [
  { id: 'jd',          label: 'JD Analyzer',      icon: Briefcase },
  { id: 'resume',      label: 'Resume Extractor',  icon: FileText  },
  { id: 'coverletter', label: 'Cover Letter',       icon: Mail      },
  { id: 'playlist',    label: 'Playlist Guide',     icon: Youtube   },
]

function SkillBadge({ skill, variant }) {
  const styles = {
    match: 'bg-green-50 text-green-700 border-green-200',
    gap:   'bg-red-50 text-red-700 border-red-200',
    win:   'bg-amber-50 text-amber-700 border-amber-200',
    plain: 'bg-gray-50 text-gray-700 border-gray-200',
  }
  return (
    <span className={clsx('inline-flex items-center text-xs px-2 py-0.5 rounded-full border', styles[variant] || styles.plain)}>
      {skill}
    </span>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button onClick={copy} className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors">
      {copied ? <Check size={13} className="text-green-600" /> : <Copy size={13} />}
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

function DownloadPdfButton({ text, filename = 'document', title = 'Document' }) {
  const download = () => {
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>${title}</title>
<style>
  body { font-family: Georgia, 'Times New Roman', serif; max-width: 720px; margin: 48px auto;
         line-height: 1.8; font-size: 13px; color: #111; padding: 0 24px; }
  pre  { white-space: pre-wrap; font-family: inherit; font-size: 13px; }
  h2   { font-size: 15px; font-weight: bold; margin-bottom: 16px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
  @media print { body { margin: 0; } }
</style>
</head><body>
<h2>${title}</h2>
<pre>${escaped}</pre>
<script>window.onload = () => { window.print(); }<\/script>
</body></html>`
    const win = window.open('', '_blank', 'width=800,height=600')
    if (!win) { alert('Allow popups to download as PDF.'); return }
    win.document.write(html)
    win.document.close()
  }
  return (
    <button onClick={download} className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors">
      <Download size={13} />
      PDF
    </button>
  )
}

function PdfUploadButton({ onExtracted, label = 'Upload PDF' }) {
  const [uploading, setUploading] = useState(false)
  const [info, setInfo]           = useState(null)
  const inputRef = useRef(null)

  const handleFile = async (file) => {
    if (!file) return
    setUploading(true); setInfo(null)
    try {
      const data = await career.parsePdf(file)
      onExtracted(data.text)
      setInfo(`${file.name} — ${data.pages} page${data.pages !== 1 ? 's' : ''}`)
    } catch (e) {
      setInfo(`Error: ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={e => handleFile(e.target.files?.[0])}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="flex items-center gap-1.5 text-xs text-accent-600 hover:text-accent-700 font-medium border border-accent-200 hover:border-accent-400 bg-accent-50 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-50"
      >
        {uploading
          ? <><Loader2 size={12} className="animate-spin" /> Extracting…</>
          : <><FileUp size={12} /> {label}</>
        }
      </button>
      {info && (
        <span className={clsx('text-xs', info.startsWith('Error') ? 'text-red-500' : 'text-gray-400')}>
          {info}
        </span>
      )}
    </div>
  )
}

export default function CareerPanel({ onNavigate }) {
  const [tab, setTab]         = useState('jd')
  const [text, setText]       = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState('')

  // Cover letter
  const [clNotes, setClNotes] = useState('')

  // Playlist guide
  const [urls, setUrls] = useState([''])

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await career.analyze(text.trim(), tab)
      if (data.error) { setError(data.error); return }
      setResult(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const generateCoverLetter = async () => {
    if (!text.trim()) return
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await career.coverLetter(text.trim(), clNotes.trim())
      if (data.error) { setError(data.error); return }
      setResult(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const generatePlaylistGuide = async () => {
    const validUrls = urls.filter(u => u.trim())
    if (!validUrls.length) return
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await career.playlistGuide(validUrls)
      if (data.error) { setError(data.error); return }
      setResult(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const switchTab = (id) => {
    setTab(id); setText(''); setResult(null); setError(''); setClNotes(''); setUrls([''])
  }

  const goToLearningPath = () => onNavigate?.('learningpath')

  const addUrl    = () => setUrls(p => [...p, ''])
  const removeUrl = (i) => setUrls(p => p.filter((_, idx) => idx !== i))
  const updateUrl = (i, v) => setUrls(p => p.map((u, idx) => idx === i ? v : u))

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Briefcase size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Career Tools</h2>
        </div>
        <p className="text-sm text-gray-500">Analyze jobs, extract skills, write cover letters, and build study guides</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-100 px-6 overflow-x-auto shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => switchTab(id)}
            className={clsx(
              'flex items-center gap-2 py-3 px-1 mr-6 text-sm font-medium border-b-2 -mb-px transition-colors whitespace-nowrap',
              tab === id ? 'border-accent-600 text-accent-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            )}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">

        {/* ── JD Analyzer ── */}
        {tab === 'jd' && (
          <>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-gray-600">Paste the job description</label>
                <PdfUploadButton onExtracted={setText} label="Upload JD PDF" />
              </div>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                rows={10}
                placeholder="Paste the full job description — responsibilities, requirements, nice-to-haves…"
                className="w-full input-field text-sm resize-y leading-relaxed min-h-[160px]"
              />
            </div>
            <button onClick={analyze} disabled={loading || !text.trim()} className="btn-primary w-full">
              {loading ? <><Loader2 size={15} className="animate-spin" /> Analyzing…</> : 'Analyze Skill Gap'}
            </button>
          </>
        )}

        {/* ── Resume Extractor ── */}
        {tab === 'resume' && (
          <>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-gray-600">Paste your resume / CV</label>
                <PdfUploadButton onExtracted={setText} label="Upload Resume PDF" />
              </div>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                rows={10}
                placeholder="Paste your resume text, LinkedIn summary, or any skills/experience description…"
                className="w-full input-field text-sm resize-y leading-relaxed min-h-[160px]"
              />
            </div>
            <button onClick={analyze} disabled={loading || !text.trim()} className="btn-primary w-full">
              {loading ? <><Loader2 size={15} className="animate-spin" /> Analyzing…</> : 'Extract Skills'}
            </button>
          </>
        )}

        {/* ── Cover Letter ── */}
        {tab === 'coverletter' && (
          <>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-gray-600">Job description</label>
                <PdfUploadButton onExtracted={setText} label="Upload JD PDF" />
              </div>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                rows={7}
                placeholder="Paste the job description you're applying to…"
                className="w-full input-field text-sm resize-y leading-relaxed min-h-[120px]"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">
                Personal notes <span className="text-gray-400 font-normal">(optional)</span>
              </label>
              <textarea
                value={clNotes}
                onChange={e => setClNotes(e.target.value)}
                rows={3}
                placeholder="e.g. 5 years Python, led a team of 4, passionate about ML…"
                className="w-full input-field text-sm resize-y leading-relaxed"
              />
            </div>
            <button onClick={generateCoverLetter} disabled={loading || !text.trim()} className="btn-primary w-full">
              {loading ? <><Loader2 size={15} className="animate-spin" /> Generating…</> : 'Generate Cover Letter'}
            </button>
          </>
        )}

        {/* ── Playlist Guide ── */}
        {tab === 'playlist' && (
          <>
            <div className="space-y-2">
              <label className="block text-xs font-medium text-gray-600 mb-1">YouTube video URLs</label>
              {urls.map((url, i) => (
                <div key={i} className="flex gap-2">
                  <input
                    value={url}
                    onChange={e => updateUrl(i, e.target.value)}
                    placeholder="https://youtube.com/watch?v=…"
                    className="input-field text-sm flex-1"
                  />
                  {urls.length > 1 && (
                    <button onClick={() => removeUrl(i)} className="text-gray-400 hover:text-red-500 transition-colors p-2">
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
              <button onClick={addUrl} className="flex items-center gap-1.5 text-xs text-accent-600 hover:text-accent-700 font-medium mt-1">
                <Plus size={13} /> Add another URL
              </button>
            </div>
            <button onClick={generatePlaylistGuide} disabled={loading || !urls.some(u => u.trim())} className="btn-primary w-full">
              {loading ? <><Loader2 size={15} className="animate-spin" /> Building guide…</> : 'Generate Study Guide'}
            </button>
          </>
        )}

        {/* ── Error ── */}
        {error && (
          <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
            <AlertCircle size={15} className="shrink-0" />
            {error}
          </div>
        )}

        {/* ── JD Results ── */}
        {result?.mode === 'jd' && (
          <div className="space-y-4">
            {result.role_title && (
              <div className="card p-4">
                <p className="text-xs font-medium text-gray-400 mb-1">Detected Role</p>
                <p className="text-base font-semibold text-gray-900">{result.role_title}</p>
              </div>
            )}
            {result.matched?.length > 0 && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle size={15} className="text-green-600" />
                  <p className="text-sm font-semibold text-gray-800">You already have ({result.matched.length})</p>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.matched.map(s => <SkillBadge key={s} skill={s} variant="match" />)}
                </div>
              </div>
            )}
            {result.gaps?.length > 0 && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-3">
                  <XCircle size={15} className="text-red-500" />
                  <p className="text-sm font-semibold text-gray-800">Skills to build ({result.gaps.length})</p>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.gaps.map(s => <SkillBadge key={s} skill={s} variant="gap" />)}
                </div>
              </div>
            )}
            {result.quick_wins?.length > 0 && (
              <div className="card p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Zap size={15} className="text-amber-500" />
                  <p className="text-sm font-semibold text-gray-800">Quick wins to focus on first</p>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.quick_wins.map(s => <SkillBadge key={s} skill={s} variant="win" />)}
                </div>
              </div>
            )}
            <button onClick={goToLearningPath} className="btn-primary w-full flex items-center justify-center gap-2">
              Build Learning Path for This Role <ChevronRight size={15} />
            </button>
          </div>
        )}

        {/* ── Resume Results ── */}
        {result?.mode === 'resume' && (
          <div className="space-y-4">
            {result.profile_updated && (
              <div className="flex items-center gap-2 bg-green-50 border border-green-200 text-green-700 text-sm px-4 py-3 rounded-lg">
                <CheckCircle size={15} />
                Skills extracted and merged into your profile automatically.
              </div>
            )}
            {result.current_role && (
              <div className="card p-4">
                <p className="text-xs font-medium text-gray-400 mb-1">Current / Most Recent Role</p>
                <p className="text-base font-semibold text-gray-900">{result.current_role}</p>
                <p className="text-xs text-gray-500 mt-0.5 capitalize">{result.experience_level} level</p>
              </div>
            )}
            {result.skills?.length > 0 && (
              <div className="card p-4">
                <p className="text-sm font-semibold text-gray-800 mb-3">Extracted Skills ({result.skills.length})</p>
                <div className="flex flex-wrap gap-1.5">
                  {result.skills.map(s => <SkillBadge key={s} skill={s} variant="match" />)}
                </div>
              </div>
            )}
            {result.suggested_roles?.length > 0 && (
              <div className="card p-4">
                <p className="text-sm font-semibold text-gray-800 mb-3">Suggested Career Paths</p>
                <div className="space-y-2">
                  {result.suggested_roles.map(role => (
                    <button
                      key={role}
                      onClick={goToLearningPath}
                      className="w-full text-left text-sm px-3 py-2 rounded-lg border border-gray-200 hover:border-accent-300 hover:bg-accent-50 transition-all flex items-center justify-between"
                    >
                      <span>{role}</span>
                      <ChevronRight size={13} className="text-gray-400" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Cover Letter Result ── */}
        {result?.cover_letter && (
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-gray-800">Cover Letter</p>
              <div className="flex items-center gap-3">
                <CopyButton text={result.cover_letter} />
                <DownloadPdfButton text={result.cover_letter} filename="cover-letter" title="Cover Letter" />
              </div>
            </div>
            <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed bg-gray-50 rounded-lg p-4 border border-gray-100">
              {result.cover_letter}
            </pre>
          </div>
        )}

        {/* ── Playlist Guide Result ── */}
        {result?.guide && (
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-gray-800">
                Study Guide
                {result.video_count != null && (
                  <span className="text-xs text-gray-400 font-normal ml-2">
                    ({result.video_count} video{result.video_count !== 1 ? 's' : ''})
                  </span>
                )}
              </p>
              <div className="flex items-center gap-3">
                <CopyButton text={result.guide} />
                <DownloadPdfButton text={result.guide} filename="study-guide" title="YouTube Playlist Study Guide" />
              </div>
            </div>
            {result.failed?.length > 0 && (
              <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 px-3 py-2 rounded-lg">
                Could not fetch transcripts for {result.failed.length} video{result.failed.length > 1 ? 's' : ''} (no captions available).
              </p>
            )}
            <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans leading-relaxed bg-gray-50 rounded-lg p-4 border border-gray-100">
              {result.guide}
            </pre>
          </div>
        )}

      </div>
    </div>
  )
}
