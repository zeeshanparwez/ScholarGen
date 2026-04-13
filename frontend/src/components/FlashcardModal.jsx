import { useState, useEffect } from 'react'
import { X, Zap, ChevronLeft, ChevronRight, RotateCcw, Loader2, CheckCircle2, XCircle } from 'lucide-react'
import { flashcards } from '../api'
import clsx from 'clsx'

export default function FlashcardsPanel() {
  const [specs, setSpecs] = useState({})
  const [selectedSpec, setSelectedSpec] = useState('')
  const [selectedSubject, setSelectedSubject] = useState('')
  const [topic, setTopic] = useState('')
  const [numQ, setNumQ] = useState(5)
  const [cards, setCards] = useState([])
  const [idx, setIdx] = useState(0)
  const [chosen, setChosen] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [phase, setPhase] = useState('form') // form | quiz | done

  useEffect(() => {
    flashcards.specializations()
      .then(d => {
        setSpecs(d.specializations || {})
        const first = Object.keys(d.specializations || {})[0]
        if (first) { setSelectedSpec(first); setSelectedSubject(d.specializations[first]?.[0] ?? '') }
      })
      .catch(() => {})
  }, [])

  const specSubjects = specs[selectedSpec] || []

  const generate = async () => {
    if (!topic.trim() || !selectedSpec || !selectedSubject) return
    setLoading(true); setError('')
    try {
      const data = await flashcards.generate(selectedSpec, selectedSubject, topic.trim(), numQ)
      setCards(data.flashcards || [])
      setIdx(0); setChosen(null); setPhase('quiz')
    } catch (err) {
      setError(err.message)
    } finally { setLoading(false) }
  }

  const pick = (i) => { if (chosen !== null) return; setChosen(i) }
  const next = () => { setIdx(i => i + 1); setChosen(null) }
  const prev = () => { setIdx(i => i - 1); setChosen(null) }
  const restart = () => { setPhase('form'); setCards([]); setTopic(''); setChosen(null); setIdx(0) }

  const card = cards[idx]

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-1">
          <Zap size={20} className="text-accent-600" strokeWidth={2} />
          <h2 className="text-lg font-semibold text-gray-900">GATE Flashcards</h2>
        </div>
        <p className="text-sm text-gray-500">AI-generated MCQs for GATE preparation</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {/* FORM */}
        {phase === 'form' && (
          <div className="max-w-lg space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Specialization</label>
              <select
                className="input-field"
                value={selectedSpec}
                onChange={e => {
                  setSelectedSpec(e.target.value)
                  setSelectedSubject(specs[e.target.value]?.[0] ?? '')
                }}
              >
                {Object.keys(specs).map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Subject</label>
              <select
                className="input-field"
                value={selectedSubject}
                onChange={e => setSelectedSubject(e.target.value)}
              >
                {specSubjects.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Topic</label>
              <input
                className="input-field"
                placeholder="e.g. Dynamic Programming, Dijkstra's algorithm…"
                value={topic}
                onChange={e => setTopic(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && generate()}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1.5">Number of Questions</label>
              <select className="input-field w-32" value={numQ} onChange={e => setNumQ(+e.target.value)}>
                {[3, 5, 10].map(n => <option key={n} value={n}>{n} questions</option>)}
              </select>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
            )}

            <button
              className="btn-primary w-full py-2.5"
              onClick={generate}
              disabled={loading || !topic.trim()}
            >
              {loading ? <><Loader2 size={16} className="animate-spin" /> Generating with Gemini…</> : <><Zap size={16} /> Generate Flashcards</>}
            </button>
          </div>
        )}

        {/* QUIZ */}
        {phase === 'quiz' && card && (
          <div className="max-w-lg">
            {/* Progress */}
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-gray-500 font-medium">{idx + 1} / {cards.length}</span>
              <button onClick={restart} className="btn-ghost text-xs py-1 px-2">
                <RotateCcw size={13} /> New set
              </button>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full mb-5">
              <div
                className="h-1.5 bg-accent-500 rounded-full transition-all"
                style={{ width: `${((idx + 1) / cards.length) * 100}%` }}
              />
            </div>

            {/* Question */}
            <div className="card p-5 mb-4">
              <p className="text-sm font-semibold text-gray-900 leading-relaxed">{card.question}</p>
            </div>

            {/* Options */}
            <div className="space-y-2 mb-4">
              {card.options.map((opt, i) => {
                const isCorrect = i === card.correct_index
                const isPicked  = i === chosen
                let cls = 'w-full text-left px-4 py-3 rounded-xl border text-sm transition-all '
                if (chosen === null) {
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
                      <span className="w-5 h-5 rounded-full border border-current flex items-center justify-center text-xs shrink-0">
                        {String.fromCharCode(65 + i)}
                      </span>
                      {opt}
                      {chosen !== null && isCorrect && <CheckCircle2 size={15} className="ml-auto text-emerald-500" />}
                      {chosen !== null && isPicked && !isCorrect && <XCircle size={15} className="ml-auto text-red-400" />}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Explanation */}
            {chosen !== null && (
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-4 text-sm text-gray-700 leading-relaxed animate-fade-in">
                <span className="font-medium text-blue-700 block mb-1">Explanation</span>
                {card.explanation}
              </div>
            )}

            {/* Navigation */}
            <div className="flex items-center gap-2">
              <button className="btn-ghost flex-1" onClick={prev} disabled={idx === 0}>
                <ChevronLeft size={16} /> Previous
              </button>
              {idx < cards.length - 1 ? (
                <button className="btn-primary flex-1" onClick={next} disabled={chosen === null}>
                  Next <ChevronRight size={16} />
                </button>
              ) : (
                <button className="btn-primary flex-1 bg-emerald-600 hover:bg-emerald-700" onClick={restart}>
                  Finish
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
