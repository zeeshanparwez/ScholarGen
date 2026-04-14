import { useState, useEffect, useRef, useCallback } from 'react'
import { Timer, Play, Pause, RotateCcw, X } from 'lucide-react'
import clsx from 'clsx'

const MODES = [
  { id: 'focus',  label: 'Focus',       minutes: 25, color: 'text-accent-600' },
  { id: 'short',  label: 'Short Break', minutes: 5,  color: 'text-emerald-600' },
  { id: 'long',   label: 'Long Break',  minutes: 15, color: 'text-blue-600' },
]

export default function PomodoroTimer({ onClose }) {
  const [modeIdx, setModeIdx]   = useState(0)
  const [seconds, setSeconds]   = useState(MODES[0].minutes * 60)
  const [running, setRunning]   = useState(false)
  const [sessions, setSessions] = useState(0)
  const intervalRef = useRef(null)
  const mode = MODES[modeIdx]

  const clear = () => { clearInterval(intervalRef.current); intervalRef.current = null }

  const tick = useCallback(() => {
    setSeconds(s => {
      if (s <= 1) {
        clear()
        setRunning(false)
        if (MODES[modeIdx].id === 'focus') setSessions(n => n + 1)
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('UpskillOS', { body: `${MODES[modeIdx].label} complete!` })
        }
        return 0
      }
      return s - 1
    })
  }, [modeIdx])

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(tick, 1000)
    } else {
      clear()
    }
    return clear
  }, [running, tick])

  const switchMode = (idx) => {
    clear(); setRunning(false)
    setModeIdx(idx)
    setSeconds(MODES[idx].minutes * 60)
  }

  const reset = () => {
    clear(); setRunning(false)
    setSeconds(mode.minutes * 60)
  }

  const toggleRun = () => {
    if (!running && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
    setRunning(r => !r)
  }

  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = (seconds % 60).toString().padStart(2, '0')
  const pct = 1 - seconds / (mode.minutes * 60)
  const r = 36, circ = 2 * Math.PI * r

  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl shadow-lg p-4 w-[220px]">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <Timer size={13} className="text-gray-400" />
          <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">Pomodoro</span>
          {sessions > 0 && (
            <span className="text-xs bg-accent-50 text-accent-700 px-1.5 rounded-full">{sessions}</span>
          )}
        </div>
        <button onClick={onClose} className="text-gray-300 hover:text-gray-500 transition-colors">
          <X size={13} />
        </button>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-1 mb-4">
        {MODES.map((m, i) => (
          <button
            key={m.id}
            onClick={() => switchMode(i)}
            className={clsx(
              'flex-1 text-[10px] py-1 rounded-lg transition-colors font-medium',
              modeIdx === i ? 'bg-gray-100 text-gray-800' : 'text-gray-400 hover:text-gray-600'
            )}
          >
            {m.id === 'focus' ? 'Focus' : m.id === 'short' ? 'Short' : 'Long'}
          </button>
        ))}
      </div>

      {/* Circular timer */}
      <div className="flex justify-center mb-4">
        <div className="relative w-20 h-20">
          <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r={r} fill="none" stroke="#f3f4f6" strokeWidth="4" />
            <circle
              cx="40" cy="40" r={r} fill="none"
              stroke={running ? '#4f46e5' : '#d1d5db'}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={circ}
              strokeDashoffset={circ * (1 - pct)}
              className="transition-all duration-1000"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={clsx('text-lg font-bold tabular-nums', mode.color)}>{m}:{s}</span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="text-gray-400 hover:text-gray-600 transition-colors" title="Reset">
          <RotateCcw size={15} />
        </button>
        <button
          onClick={toggleRun}
          className={clsx(
            'w-9 h-9 rounded-full flex items-center justify-center transition-all',
            running ? 'bg-red-100 text-red-600 hover:bg-red-200' : 'bg-accent-600 text-white hover:bg-accent-700'
          )}
        >
          {running ? <Pause size={15} /> : <Play size={15} />}
        </button>
      </div>
    </div>
  )
}
