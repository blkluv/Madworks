import { cn } from "@/lib/utils"
import { format } from "date-fns"
import { User, Bot } from "lucide-react"

type Attachment = {
  type: 'image'
  url: string
  href?: string
  variant?: string
}

type MessageProps = {
  id: string
  role: 'user' | 'assistant'
  content?: string
  attachments?: Attachment[]
  timestamp: string
  isGrouped?: boolean
}

export function Message({ role, content, attachments, timestamp, isGrouped = false }: MessageProps) {
  const isUser = role === 'user'
  const hasContent = !!content && content.trim().length > 0
  const hasAttachments = !!attachments && attachments.length > 0
  
  const formatTime = (dateString: string) => {
    return format(new Date(dateString), 'h:mm a')
  }

  const renderAttachments = () => {
    if (!attachments?.length) return null

    const count = attachments.length
    
    if (count === 1) {
      const img = attachments[0]
      return (
        <a 
          href={img.href || img.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="block rounded-lg overflow-hidden border border-zinc-700 hover:border-zinc-500 transition-colors relative"
        >
          <img 
            src={img.url} 
            alt="Attachment"
            className="w-full h-48 md:h-64 lg:h-72 object-cover block"
          />
          {img.variant && (
            <span className="absolute top-2 left-2 bg-black/70 text-xs px-2 py-0.5 rounded-md border border-white/10">
              {img.variant}
            </span>
          )}
        </a>
      )
    }

    // For multiple images, show a grid
    return (
      <div className={cn(
        "grid gap-2",
        count <= 2 ? "grid-cols-2" : "grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
      )}>
        {attachments.map((img, i) => (
          <a 
            key={i}
            href={img.href || img.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="block rounded-md overflow-hidden border border-zinc-700 hover:border-zinc-500 transition-colors relative"
          >
            <img 
              src={img.url} 
              alt={`Attachment ${i + 1}`}
              className="w-full h-28 sm:h-32 md:h-36 lg:h-40 object-cover block"
            />
            {img.variant && (
              <span className="absolute top-1.5 left-1.5 bg-black/70 text-[10px] px-1.5 py-[2px] rounded border border-white/10">
                {img.variant}
              </span>
            )}
          </a>
        ))}
      </div>
    )
  }

  return (
    <div className={cn(
      "group flex gap-3 p-4",
      isUser ? 'justify-end' : 'justify-start',
      !isGrouped && (isUser ? 'bg-zinc-900/50' : 'bg-zinc-800/50')
    )}>
      {!isUser && !isGrouped && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center">
          <Bot className="w-4 h-4 text-white" />
        </div>
      )}
      
      <div className={cn(
        "flex-1 max-w-3xl",
        isUser && 'flex flex-col items-end'
      )}>
        {!isGrouped && (
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium">
              {isUser ? 'You' : 'Assistant'}
            </span>
            <span className="text-xs text-zinc-500">
              {formatTime(timestamp)}
            </span>
          </div>
        )}
        
        <div className={cn(
          "rounded-xl text-sm",
          hasContent ? 'p-3' : 'p-0',
          isUser 
            ? 'bg-black/50 border border-zinc-700/60' 
            : 'bg-zinc-700/40 border border-zinc-700/50'
        )}>
          {hasContent && (
            <div className="prose prose-invert prose-sm max-w-none">
              {content.split('\n').map((line, i) => (
                <p key={i}>{line || <br />}</p>
              ))}
            </div>
          )}
          {hasAttachments && (
            <div className={cn(hasContent ? "-mx-3 -mb-3 mt-2" : undefined)}>
              {renderAttachments()}
            </div>
          )}
        </div>
      </div>
      
      {isUser && !isGrouped && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center">
          <User className="w-4 h-4 text-zinc-300" />
        </div>
      )}
    </div>
  )
}

type MessageGroupProps = {
  date: string
  messages: MessageProps[]
}

export function MessageGroup({ date, messages }: MessageGroupProps) {
  const formattedDate = new Date(date).toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  })

  return (
    <div className="space-y-1">
      <div className="sticky top-0 z-10 py-2 bg-zinc-900/80 backdrop-blur-sm">
        <div className="text-center">
          <span className="inline-block px-3 py-1 text-xs font-medium text-zinc-400 bg-zinc-800/80 rounded-full">
            {formattedDate}
          </span>
        </div>
      </div>
      
      <div className="space-y-1">
        {messages.map((message, i) => (
          <Message
            key={message.id}
            {...message}
            isGrouped={i > 0 && messages[i-1].role === message.role}
          />
        ))}
      </div>
    </div>
  )
}
