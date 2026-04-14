import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { GraduationCap, Volume2, VolumeX, Bookmark, BookmarkCheck } from 'lucide-react'
import clsx from 'clsx'
import { bookmarks } from '../api'

function useSpeech() {
  const [speaking, setSpeaking] = useState(false)

  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return
    if (speaking) {
      window.speechSynthesis.cancel()
      setSpeaking(false)
      return
    }
    const plain = text
      .replace(/[*_`#>~]/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/!\[[^\]]*\]\([^)]+\)/g, '')
      .replace(/\n{2,}/g, '. ')
      .replace(/\n/g, ' ')
      .trim()
    const utt = new SpeechSynthesisUtterance(plain)
    utt.rate = 1.0
    utt.onend = () => setSpeaking(false)
    utt.onerror = () => setSpeaking(false)
    setSpeaking(true)
    window.speechSynthesis.speak(utt)
  }, [speaking])

  return { speaking, speak }
}

export default function ChatMessage({ role, content, streaming }) {
  const isUser = role === 'user'
  const { speaking, speak } = useSpeech()
  const [bookmarked, setBookmarked] = useState(false)
  const [bookmarking, setBookmarking] = useState(false)

  const saveBookmark = useCallback(async () => {
    if (bookmarked || bookmarking || !content) return
    setBookmarking(true)
    try {
      await bookmarks.add(content)
      setBookmarked(true)
    } catch { /* non-critical */ }
    finally { setBookmarking(false) }
  }, [bookmarked, bookmarking, content])

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

        {/* Action buttons — assistant only, on hover, hidden while streaming */}
        {!isUser && !streaming && content && (
          <div className="mt-1 ml-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {/* Speak */}
            <button
              onClick={() => speak(content)}
              title={speaking ? 'Stop speaking' : 'Read aloud'}
              className={clsx(
                'flex items-center gap-1 text-xs px-2 py-0.5 rounded-full transition-all',
                speaking
                  ? 'bg-accent-100 text-accent-700 opacity-100'
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
              )}
            >
              {speaking ? <VolumeX size={11} /> : <Volume2 size={11} />}
              <span>{speaking ? 'Stop' : 'Speak'}</span>
            </button>

            {/* Bookmark */}
            <button
              onClick={saveBookmark}
              disabled={bookmarked || bookmarking}
              title={bookmarked ? 'Bookmarked' : 'Save to bookmarks'}
              className={clsx(
                'flex items-center gap-1 text-xs px-2 py-0.5 rounded-full transition-all',
                bookmarked
                  ? 'bg-amber-50 text-amber-600 opacity-100'
                  : 'text-gray-400 hover:text-amber-500 hover:bg-amber-50'
              )}
            >
              {bookmarked ? <BookmarkCheck size={11} /> : <Bookmark size={11} />}
              <span>{bookmarked ? 'Saved' : 'Save'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
