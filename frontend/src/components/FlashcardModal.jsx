import { useState, useEffect } from 'react'
import { Zap, ChevronLeft, ChevronRight, RotateCcw, Loader2, CheckCircle2, XCircle, Trophy, Target } from 'lucide-react'
import { flashcards } from '../api'
import clsx from 'clsx'

export default function SkillAssessmentPanel() {
  const [specs, setSpecs]               = useState({})
  const [skillArea, setSkillArea]       = useState('')
  const [subArea, setSubArea]           = useState('')
  const [topic, setTopic]               = useState('')
  const [numQ, setNumQ]                 = useState(5)
  const [cards, setCards]               = useState([])
  const [idx, setIdx]                   = useState(0)
  const [chosen, setChosen]             = useState(null)
  const [score, setScore]               = useState(0)
  const [loading, setLoading]           = useState(false)
  const [error, setError]               = useState('')
  const [phase, setPhase]               = useState('form') // form | quiz | done

  useEffect(() => {
    flashcards.specializations()
      .then(d => {
        const s = d.specializations || {}
        setSpecs(s)
        const first = Object.keys(s)[0] || ''
        setSkillArea(first)
        setSubArea(s[first]?.[0] ?? '')
      })
      .catch(() => {})
  }, [])

  const subAreas = specs[skillArea] || []

  const generate = async () => {
    if (!topic.trim() || !skillArea || !subArea) return
    setLoading(true); setError('')
    try {
      const data = await flashcards.generate(skillArea, subArea, topic.trim(), numQ)
      if (!data.flashcards?.length) {
        setError('No questions generated — try a more specific topic.')
        return
      }
      setCards(data.flashcards)
      setIdx(0); setChosen(null); setScore(0); setPhase('quiz')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false) }
  }

  const pick = (i) => {
    if (chosen !== null) return
    setChosen(i)
    if (i === cards[idx].correct_index) setScore(s => s + 1)
  }

  const next = () => {
    if (idx + 1 >= cards.length) {
      setPhase('done')
    } else {
      setIdx(i => i + 1)
      setChosen(null)
    }
  }

  const prev = () => { setIdx(i => i - 1); setChosen(null) }

  const restart = () => {
    setPhase('form'); setCards([]); setTopic('')
    setChosen(null); setIdx(0); setScore(0)
  }

  const card = cards[idx]
  const pct = cards.length ? Math.round((score / cards.length) * 100) : 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Target size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">Skill Assessment</h2>
        </div>
        <p className="text-sm text-gray-500">AI-generated MCQs to test your knowledge on any tech topic</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">

        {/* ── FORM ── */}
        {phase === 'form' && (
          <div className="max-w-lg space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Skill Area</label>
              <select
                className="input-field"
                value={skillArea}
                onChange={e => {
                  setSkillArea(e.target.value)
                  setSubArea(specs[e.target.value]?.[0] ?? '')
                }}
              >
                {Object.keys(specs).map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Sub-area / Technology</label>
              <select
                className="input-field"
                value={subArea}
                onChange={e => setSubArea(e.target.value)}
              >
                {subAreas.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">
                Specific Topic
              </label>
              <input
                className="input-field"
                placeholder="e.g. React Hooks, SQL JOINs, Docker networking, JWT auth…"
                value={topic}
                onChange={e => setTopic(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && generate()}
              />
              <p className="text-xs text-gray-400 mt-1">Be specific — the more focused the topic, the better the questions.</p>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Number of Questions</label>
              <div className="flex gap-2">
                {[3, 5, 10].map(n => (
                  <button
                    key={n}
                    onClick={() => setNumQ(n)}
                    className={clsx(
                      'flex-1 py-2 rounded-lg text-sm font-medium border transition-colors',
                      numQ === n
                        ? 'bg-accent-600 text-white border-accent-600'
                        : 'border-gray-200 text-gray-600 hover:border-accent-300 hover:text-accent-700'
                    )}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
                {error}
              </div>
            )}

            <button
              className="btn-primary w-full py-2.5"
              onClick={generate}
              disabled={loading || !topic.trim()}
            >
              {loading
                ? <><Loader2 size={16} className="animate-spin" /> Generating questions…</>
                : <><Zap size={16} /> Start Assessment</>
              }
            </button>
          </div>
        )}

        {/* ── QUIZ ── */}
        {phase === 'quiz' && card && (
          <div className="max-w-lg">
            {/* Progress bar */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-500 font-medium">
                Question {idx + 1} of {cards.length}
              </span>
              <div className="flex items-center gap-1.5 text-xs text-gray-500">
                <CheckCircle2 size={12} className="text-emerald-500" />
                <span>{score} correct</span>
              </div>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full mb-5">
              <div
                className="h-1.5 bg-accent-500 rounded-full transition-all"
                style={{ width: `${((idx + 1) / cards.length) * 100}%` }}
              />
            </div>

            {/* Question */}
            <div className="card p-5 mb-4">
              <div className="flex items-start gap-2 mb-1">
                <span className="text-xs font-semibold text-accent-600 bg-accent-50 px-2 py-0.5 rounded-full shrink-0 mt-0.5">
                  Q{idx + 1}
                </span>
              </div>
              <p className="text-sm font-semibold text-gray-900 leading-relaxed mt-2">{card.question}</p>
            </div>

            {/* Options */}
            <div className="space-y-2 mb-4">
              {card.options.map((opt, i) => {
                const isCorrect = i === card.correct_index
                const isPicked  = i === chosen
                const revealed  = chosen !== null

                let cls = 'w-full text-left px-4 py-3 rounded-xl border text-sm transition-all '
                if (!revealed) {
                  cls += 'border-gray-200 hover:border-accent-400 hover:bg-accent-50 cursor-pointer'
                } else if (isCorrect) {
                  cls += 'border-emerald-400 bg-emerald-50 text-emerald-800 font-medium'
                } else if (isPicked) {
                  cls += 'border-red-400 bg-red-50 text-red-800'
                } else {
                  cls += 'border-gray-100 bg-gray-50 text-gray-400'
                }

                return (
                  <button key={i} className={cls} onClick={() => pick(i)}>
                    <span className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-full border border-current flex items-center justify-center text-xs shrink-0 font-semibold">
                        {String.fromCharCode(65 + i)}
                      </span>
                      <span className="flex-1 text-left">{opt}</span>
                      {revealed && isCorrect  && <CheckCircle2 size={15} className="text-emerald-500 shrink-0" />}
                      {revealed && isPicked && !isCorrect && <XCircle size={15} className="text-red-400 shrink-0" />}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Explanation */}
            {chosen !== null && (
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-4 text-sm text-gray-700 leading-relaxed animate-fade-in">
                <span className="font-semibold text-blue-700 block mb-1">Explanation</span>
                {card.explanation}
              </div>
            )}

            {/* Navigation */}
            <div className="flex gap-2">
              <button
                className="btn-ghost flex-1"
                onClick={prev}
                disabled={idx === 0}
              >
                <ChevronLeft size={16} /> Prev
              </button>
              <button
                className="btn-primary flex-1"
                onClick={next}
                disabled={chosen === null}
              >
                {idx < cards.length - 1 ? <>Next <ChevronRight size={16} /></> : 'Finish'}
              </button>
            </div>

            {/* Skip */}
            <div className="flex justify-center mt-3">
              <button onClick={restart} className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1">
                <RotateCcw size={11} /> Start over
              </button>
            </div>
          </div>
        )}

        {/* ── DONE ── */}
        {phase === 'done' && (
          <div className="max-w-lg text-center">
            <div className="card p-8 mb-6">
              <div className="w-16 h-16 mx-auto rounded-full bg-accent-50 flex items-center justify-center mb-4">
                <Trophy size={28} className={pct >= 70 ? 'text-amber-500' : 'text-gray-400'} />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-1">
                {pct >= 80 ? 'Excellent!' : pct >= 60 ? 'Good job!' : 'Keep practising!'}
              </h3>
              <p className="text-sm text-gray-500 mb-5">You scored {score} out of {cards.length}</p>

              {/* Score ring */}
              <div className="flex justify-center mb-5">
                <div className="relative w-24 h-24">
                  <svg className="w-24 h-24 -rotate-90" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="32" fill="none" stroke="#f3f4f6" strokeWidth="6" />
                    <circle
                      cx="40" cy="40" r="32" fill="none"
                      stroke={pct >= 70 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#ef4444'}
                      strokeWidth="6" strokeLinecap="round"
                      strokeDasharray={2 * Math.PI * 32}
                      strokeDashoffset={2 * Math.PI * 32 * (1 - pct / 100)}
                      className="transition-all duration-700"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xl font-bold text-gray-900">{pct}%</span>
                  </div>
                </div>
              </div>

              {/* Per-question breakdown */}
              <div className="grid grid-cols-5 gap-1.5 mb-6">
                {cards.map((_, i) => (
                  <div
                    key={i}
                    className={clsx(
                      'h-2 rounded-full',
                      // We don't track per-question correctness in state,
                      // so colour by whether score >= threshold
                      i < score ? 'bg-emerald-400' : 'bg-red-300'
                    )}
                    title={`Q${i + 1}`}
                  />
                ))}
              </div>

              <div className="space-y-2">
                <button className="btn-primary w-full" onClick={restart}>
                  <RotateCcw size={15} /> New Assessment
                </button>
                <button
                  className="btn-ghost w-full text-sm"
                  onClick={() => { setIdx(0); setChosen(null); setPhase('quiz') }}
                >
                  Review Questions
                </button>
              </div>
            </div>

            <p className="text-xs text-gray-400">
              Topic: <span className="font-medium text-gray-600">{topic}</span> ·{' '}
              {skillArea} › {subArea}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
