import { useRef, useEffect } from 'react'
import { SendHorizontal, Square } from 'lucide-react'
import clsx from 'clsx'

export default function ChatInput({ onSend, disabled, streaming }) {
  const ref = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  })

  const submit = () => {
    const val = ref.current?.value.trim()
    if (!val || disabled) return
    onSend(val)
    ref.current.value = ''
    ref.current.style.height = 'auto'
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="max-w-3xl mx-auto">
        <div className={clsx(
          'flex items-end gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm transition-shadow',
          !disabled && 'focus-within:ring-2 focus-within:ring-accent-500/20 focus-within:border-accent-400'
        )}>
          <textarea
            ref={ref}
            rows={1}
            onKeyDown={onKey}
            disabled={disabled}
            placeholder={streaming ? 'Generating response…' : 'Ask anything about learning…'}
            className="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder:text-gray-400 focus:outline-none leading-6 max-h-40 min-h-[24px] disabled:opacity-60"
          />
          <button
            onClick={submit}
            disabled={disabled}
            className={clsx(
              'w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0',
              disabled
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-accent-600 text-white hover:bg-accent-700 active:scale-95'
            )}
            title={streaming ? 'Generating…' : 'Send (Enter)'}
          >
            {streaming ? <Square size={14} strokeWidth={2.5} /> : <SendHorizontal size={14} strokeWidth={2.5} />}
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </div>
  )
}
