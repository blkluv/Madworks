import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, PlusCircle, Trash2, Pencil, ChevronLeft, ChevronRight } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

type Conversation = {
  id: string
  title: string
  preview?: string
  updatedAt: string
  messageCount: number
}

type ConversationListProps = {
  conversations: Conversation[]
  activeId: string
  searchQuery: string
  onSearch: (query: string) => void
  onSelect: (id: string) => void
  onNewChat: () => void
  onRename: (id: string) => void
  onDelete: (id: string) => void
  isCollapsed: boolean
  onToggleCollapse: () => void
}

export function ConversationList({
  conversations,
  activeId,
  searchQuery,
  onSearch,
  onSelect,
  onNewChat,
  onRename,
  onDelete,
  isCollapsed,
  onToggleCollapse,
}: ConversationListProps) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
    
    if (diffInDays === 0) return 'Today'
    if (diffInDays === 1) return 'Yesterday'
    if (diffInDays < 7) return `${diffInDays} days ago`
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  return (
    <div className={cn(
      "relative flex flex-col h-full bg-zinc-900/80 border-r border-white/10 transition-all duration-300",
      isCollapsed ? "w-16" : "w-64 lg:w-80"
    )}>
      {isCollapsed ? (
        <div className="p-2 flex flex-col items-center">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapse}
            className="w-10 h-10 rounded-lg"
          >
            <ChevronRight className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onNewChat}
            className="mt-4 w-10 h-10 rounded-lg"
            title="New Chat"
          >
            <PlusCircle className="w-5 h-5" />
          </Button>
        </div>
      ) : (
        <>
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Chats</h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggleCollapse}
                className="w-8 h-8 rounded-lg"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
            </div>
            
            <div className="relative">
              <Search className="w-4 h-4 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <Input
                value={searchQuery}
                onChange={(e) => onSearch(e.target.value)}
                placeholder="Search chats..."
                className="pl-9 bg-zinc-800/50 border-zinc-700 focus-visible:ring-1 focus-visible:ring-zinc-600"
              />
            </div>
            
            <Button
              onClick={onNewChat}
              className="w-full bg-black hover:bg-zinc-900 text-white border border-zinc-700"
            >
              <PlusCircle className="w-4 h-4 mr-2" />
              New Chat
            </Button>
          </div>
          
          <ScrollArea className="flex-1 px-2">
            <div className="space-y-1 pb-4">
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => onSelect(conv.id)}
                  className={cn(
                    "group relative flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors",
                    activeId === conv.id 
                      ? "bg-zinc-800/70" 
                      : "hover:bg-zinc-800/40"
                  )}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium truncate">
                        {conv.title || "Untitled"}
                      </p>
                      <span className="text-xs text-zinc-500 whitespace-nowrap">
                        {formatDate(conv.updatedAt)}
                      </span>
                    </div>
                    {conv.preview && (
                      <p className="text-xs text-zinc-400 truncate mt-0.5">
                        {conv.preview}
                      </p>
                    )}
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-zinc-500">
                        {conv.messageCount} {conv.messageCount === 1 ? 'message' : 'messages'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onRename(conv.id)
                      }}
                      className="p-1 rounded-md hover:bg-zinc-700/50 text-zinc-400 hover:text-zinc-200"
                      title="Rename"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onDelete(conv.id)
                      }}
                      className="p-1 rounded-md hover:bg-red-900/50 text-zinc-400 hover:text-red-400"
                      title="Delete"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
              
              {conversations.length === 0 && (
                <div className="text-center py-8">
                  <p className="text-sm text-zinc-400">No conversations yet</p>
                  <Button
                    onClick={onNewChat}
                    variant="link"
                    className="text-zinc-300 hover:text-zinc-100 mt-2"
                  >
                    Start a new chat
                  </Button>
                </div>
              )}
            </div>
          </ScrollArea>
        </>
      )}
    </div>
  )
}
