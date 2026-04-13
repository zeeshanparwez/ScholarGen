import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { GraduationCap } from 'lucide-react'
import clsx from 'clsx'

export default function ChatMessage({ role, content, streaming }) {
  const isUser = role === 'user'

  return (
    <div className={clsx('flex gap-3 px-4 py-3 group', isUser && 'flex-row-reverse')}>
      {/* Avatar */}
      <div className={clsx(
        'w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 mt-0.5',
        isUser ? 'bg-accent-600 text-white' : 'bg-gray-100 text-gray-600 border border-gray-200'
      )}>
        {isUser ? (
          (localStorage.getItem('sg_user')?.[0] ?? 'U').toUpperCase()
        ) : (
          <GraduationCap size={14} strokeWidth={2} />
        )}
      </div>

      {/* Bubble */}
      <div className={clsx('max-w-[80%] min-w-0', isUser && 'items-end')}>
        <div className={clsx(
          'rounded-2xl px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-accent-600 text-white rounded-tr-sm'
            : 'bg-gray-50 text-gray-800 border border-gray-100 rounded-tl-sm'
        )}>
          {isUser ? (
            <span className="whitespace-pre-wrap">{content}</span>
          ) : (
            <div className={clsx('prose-chat', streaming && !content && 'streaming-cursor')}>
              {content ? (
                <span className={clsx(streaming && 'streaming-cursor')}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </span>
              ) : (
                <span className="text-gray-400 italic text-xs">Thinking…</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
