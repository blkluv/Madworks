import { useState, useRef, useCallback, ChangeEvent } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Paperclip, Image as ImageIcon, X, Send, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"

type AspectRatio = '1:1' | '16:9' | '9:16' | '4:3' | '3:4' | '2:3' | '3:2' | 'custom'

interface AspectRatioOption {
  value: AspectRatio
  label: string
  icon: React.ReactNode
}

const aspectRatioOptions: AspectRatioOption[] = [
  { value: '1:1', label: 'Square', icon: <span className="w-4 h-4 border border-zinc-400 rounded-sm" /> },
  { value: '16:9', label: 'Wide', icon: <span className="w-6 h-4 border border-zinc-400 rounded-sm" /> },
  { value: '9:16', label: 'Portrait', icon: <span className="w-4 h-6 border border-zinc-400 rounded-sm rotate-90" /> },
  { value: '4:3', label: 'Standard', icon: <span className="w-5 h-4 border border-zinc-400 rounded-sm" /> },
  { value: '3:4', label: 'Classic', icon: <span className="w-4 h-5 border border-zinc-400 rounded-sm rotate-90" /> },
  { value: 'custom', label: 'Custom', icon: <span className="w-4 h-4 border border-dashed border-zinc-400 rounded-sm" /> },
]

type Attachment = {
  id: string
  file: File
  preview: string
  aspectRatio: AspectRatio
}

type ComposerProps = {
  value: string
  onChange: (value: string) => void
  onSubmit: (message: string, attachments: File[]) => void
  onAspectRatioChange?: (ratio: AspectRatio) => void
  isSubmitting: boolean
  className?: string
}

export function Composer({
  value,
  onChange,
  onSubmit,
  onAspectRatioChange,
  isSubmitting,
  className,
}: ComposerProps) {
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [showAspectRatio, setShowAspectRatio] = useState(false)
  const [selectedAspectRatio, setSelectedAspectRatio] = useState<AspectRatio>('1:1')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if ((!value.trim() && attachments.length === 0) || isSubmitting) return
    
    onSubmit(value, attachments.map(a => a.file))
    setAttachments([])
    onChange('')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    const newAttachments = files.map(file => ({
      id: URL.createObjectURL(file),
      file,
      preview: URL.createObjectURL(file),
      aspectRatio: selectedAspectRatio
    }))

    setAttachments(prev => [...prev, ...newAttachments])
    setShowAspectRatio(true)
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeAttachment = (id: string) => {
    setAttachments(prev => {
      const newAttachments = prev.filter(a => a.id !== id)
      if (newAttachments.length === 0) {
        setShowAspectRatio(false)
      }
      return newAttachments
    })
  }

  const selectAspectRatio = (ratio: AspectRatio) => {
    setSelectedAspectRatio(ratio)
    onAspectRatioChange?.(ratio)
  }

  return (
    <div className={cn("w-full max-w-4xl mx-auto px-4 pb-6 pt-2", className)}>
      {attachments.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map(attachment => (
            <div key={attachment.id} className="relative group">
              <div className="w-16 h-16 rounded-md overflow-hidden bg-zinc-800 border border-zinc-700">
                <img
                  src={attachment.preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <button
                type="button"
                onClick={() => removeAttachment(attachment.id)}
                className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-red-600 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3 text-white" />
              </button>
            </div>
          ))}
        </div>
      )}

      {showAspectRatio && (
        <div className="mb-3 p-2 bg-zinc-800/50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-zinc-300">Aspect Ratio:</span>
            <div className="flex-1 flex items-center gap-1 overflow-x-auto pb-1">
              {aspectRatioOptions.map(option => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => selectAspectRatio(option.value)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs whitespace-nowrap",
                    selectedAspectRatio === option.value 
                      ? "bg-zinc-700 text-white" 
                      : "text-zinc-400 hover:bg-zinc-700/50"
                  )}
                >
                  <span className="text-zinc-400">{option.icon}</span>
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            className="min-h-[60px] max-h-[200px] w-full resize-none pr-16 py-3 bg-zinc-800/50 border-zinc-700 focus-visible:ring-1 focus-visible:ring-zinc-600"
            rows={1}
          />
          
          <div className="absolute right-2 bottom-2 flex items-center gap-1">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="w-8 h-8 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-700/50"
              onClick={() => fileInputRef.current?.click()}
              title="Attach image"
            >
              <Paperclip className="w-4 h-4" />
            </Button>
            
            <Button
              type="submit"
              size="icon"
              disabled={!value.trim() && attachments.length === 0 || isSubmitting}
              className={cn(
                "w-8 h-8 rounded-full transition-all",
                (value.trim() || attachments.length > 0) && !isSubmitting
                  ? "bg-black hover:bg-zinc-900 text-white border border-zinc-700"
                  : "bg-zinc-700 text-zinc-500"
              )}
              title="Send message"
            >
              {isSubmitting ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
        
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-xs text-zinc-400 hover:text-white h-7 px-2"
              onClick={() => {}}
            >
              <Sparkles className="w-3.5 h-3.5 mr-1" />
              Enhance with AI
            </Button>
          </div>
          
          <div className="text-xs text-zinc-500">
            {value.length}/1000
          </div>
        </div>
      </form>
    </div>
  )
}
