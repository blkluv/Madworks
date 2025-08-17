import { cn } from "@/lib/utils"
import { format } from "date-fns"
import { ImageIcon, User, Bot } from "lucide-react"
import Image from "next/image"

type Attachment = {
  type: 'image'
  url: string
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
          href={img.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="block mt-2 rounded-lg overflow-hidden border border-zinc-700 hover:border-zinc-500 transition-colors"
        >
          <img 
            src={img.url} 
            alt="Attachment"
            className="w-full max-h-[500px] object-contain bg-black/30"
          />
        </a>
      )
    }

    // For multiple images, show a grid
    return (
      <div className={cn(
        "grid gap-2 mt-2",
        count <= 2 ? "grid-cols-2" : "grid-cols-2 sm:grid-cols-3"
      )}>
        {attachments.map((img, i) => (
          <a 
            key={i}
            href={img.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="block rounded-md overflow-hidden border border-zinc-700 hover:border-zinc-500 transition-colors aspect-square"
          >
            <img 
              src={img.url} 
              alt={`Attachment ${i + 1}`}
              className="w-full h-full object-cover"
            />
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
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center">
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
          "rounded-xl p-3 text-sm",
          isUser 
            ? 'bg-indigo-600/20 border border-indigo-500/30' 
            : 'bg-zinc-700/50 border border-zinc-700/50'
        )}>
          {content && (
            <div className="prose prose-invert prose-sm max-w-none">
              {content.split('\n').map((line, i) => (
                <p key={i}>{line || <br />}</p>
              ))}
            </div>
          )}
          {renderAttachments()}
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
