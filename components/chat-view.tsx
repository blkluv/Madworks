"use client"

import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { PlusCircle, Paperclip, Image as ImageIcon, Send, SlidersHorizontal } from "lucide-react"
import { useApp } from "@/components/app-context"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  attachments?: Array<{ type: string; url: string }>
}

// Prefer a single best format per size/variant to avoid duplicates in the collage
function dedupePreferredImages(
  outs: Array<{ format: string; width: number; height: number; url: string; variant?: string }> = []
) {
  const rank: Record<string, number> = { png: 4, jpg: 3, jpeg: 3, webp: 2, svg: 1 }
  const bestByKey = new Map<string, { format: string; width: number; height: number; url: string; variant?: string }>()
  for (const o of outs) {
    const fmt = (o.format || "").toLowerCase()
    if (!["png", "jpg", "jpeg", "webp"].includes(fmt)) continue
    const key = `${o.width}x${o.height}_${o.variant || ""}`
    const existing = bestByKey.get(key)
    if (!existing || (rank[fmt] || 0) > (rank[(existing.format || "").toLowerCase()] || 0)) {
      bestByKey.set(key, o)
    }
  }
  return Array.from(bestByKey.values()).map((o) => ({ ...o, url: normalizeUrl(o.url) }))
}

// Derive a human-friendly title for the conversation from AI response
function deriveTitleFromResponse(json: PipelineResponse, fallbackPrompt: string): string {
  let t = json?.copy_best?.headline
    || json?.copy_variants?.[0]?.headline
    || (fallbackPrompt || "New chat")
  // cleanup
  t = (t || "New chat").replace(/^\s+|\s+$/g, "").replace(/[\r\n]+/g, " ")
  // keep it short
  if (t.length > 60) t = t.slice(0, 57) + "…"
  return t
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
  analysis?: any
}

type PipelineResponse = {
  copy_best?: {
    headline: string
    subheadline: string
    cta: string
  }
  copy_variants?: Array<{
    headline: string
    subheadline: string
    cta: string
  }>
  thumbnail_url?: string
  outputs?: Array<{
    format: string
    width: number
    height: number
    url: string
  }>
  error?: string
  analysis?: any
}

function normalizeUrl(u: string): string {
  if (!u) return u
  try {
    let v = u.trim()
    // Attempt to decode percent-encoded URLs like http%3A//...
    if (/%[0-9A-Fa-f]{2}/.test(v)) {
      try { v = decodeURIComponent(v) } catch {}
    }
    // If already absolute after decoding, return as-is
    if (v.startsWith("http://") || v.startsWith("https://")) return v

    // Otherwise prefix with pipeline base
    const base = (process.env.NEXT_PUBLIC_PIPELINE_BASE || "http://localhost:8010").replace(/\/$/, "")
    const path = v.startsWith("/") ? v : `/${v}`
    return `${base}${path}`
  } catch {
    return u
  }
}

export function ChatView() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const autoRanRef = useRef(false)
  const { setPendingPrompt, pendingPrompt, pendingFiles, setPendingFiles, decrementCredit } = useApp()
  const [conversations, setConversations] = useState<Conversation[]>([
    { 
      id: "c_default", 
      title: "New chat", 
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    },
  ])
  const [activeId, setActiveId] = useState<string>("c_default")
  const [isGenerating, setIsGenerating] = useState(false)
  const [input, setInput] = useState("")
  const [selectedAspectRatio, setSelectedAspectRatio] = useState("1:1")
  const [platform, setPlatform] = useState<"" | "instagram" | "facebook" | "linkedin" | "x">("")
  const [tone, setTone] = useState<"" | "bold" | "friendly" | "professional" | "minimal">("")
  const [variants, setVariants] = useState<number>(3)
  const [recommendation, setRecommendation] = useState<{
    headline: string
    subheadline: string
    cta: string
  } | null>(null)
  const [lastOutputs, setLastOutputs] = useState<Array<{ format: string; width: number; height: number; url: string; variant?: string }>>([])
  const [copyList, setCopyList] = useState<Array<{ headline: string; subheadline: string; cta: string }>>([])
  const [showOptions, setShowOptions] = useState(false)

  const suggestions: Array<{ label: string; text: string; preset?: { platform?: string; tone?: string; aspect?: string; variants?: number } }> = [
    {
      label: "Stories variants",
      text: "Try a few bold, high-contrast Story options. Keep headlines punchy and legible.",
      preset: { platform: "instagram", tone: "bold", aspect: "9:16", variants: 5 },
    },
    {
      label: "Credible LinkedIn",
      text: "Professional, concise tone. Suggest a clean layout and clear CTA.",
      preset: { platform: "linkedin", tone: "professional", aspect: "1:1", variants: 3 },
    },
    {
      label: "Seasonal promo",
      text: "Gently highlight a limited-time offer. Keep copy brand-safe and clear.",
    },
    {
      label: "Square carousel",
      text: "Simple square carousel tips. Center key text and avoid clutter.",
    },
    {
      label: "Font pairing ideas",
      text: "Suggest a couple of safe headline/body pairings from the approved set.",
    },
  ]

  const dimensionOptions = [
    { value: "1:1", label: "Square (1:1)", desc: "Instagram Feed, Carousels" },
    { value: "4:5", label: "Portrait (4:5)", desc: "IG Portrait, FB Feed" },
    { value: "9:16", label: "Story (9:16)", desc: "IG Stories/Reels, TikTok" },
    { value: "16:9", label: "Landscape (16:9)", desc: "YouTube, Web Hero" },
  ]

  // Detect when the user likely intended to reference/attach an image
  const imageIntentRegex = /(\b(this|that|the|my)\s+(image|picture|photo|screenshot|pic)\b)|(\battach(ed)?\b)|(\b(upload|use)\s+(this|the)\s+(image|picture|photo|screenshot|pic)\b)|(\b\w+\.(png|jpe?g|webp|gif)\b)/i
  const mentionsImage = (text: string) => imageIntentRegex.test(text || "")
  const hasPreviousImage = (conv: Conversation) => !!(conv?.analysis || conv?.messages?.some(m => m.attachments?.some(a => a.type === 'image')))

  const applySuggestion = (s: string) => {
    setInput((prev) => {
      if (!prev) return s
      const needsSpace = /[.!?]$/.test(prev.trim())
      return prev.trimEnd() + (needsSpace ? " " : ". ") + s
    })
  }

  const sizesForAspect = (ar: string) => {
    switch (ar) {
      case "4:5":
        return [{ name: "4:5", width: 1080, height: 1350 }]
      case "9:16":
        return [{ name: "9:16", width: 1080, height: 1920 }]
      case "16:9":
        return [{ name: "16:9", width: 1920, height: 1080 }]
      case "1:1":
      default:
        return [{ name: "1:1", width: 1080, height: 1080 }]
    }
  }

  const activeConv = conversations.find((c) => c.id === activeId) || {
    id: "c_default",
    title: "New chat",
    messages: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }

  // Track the most recent assistant message to anchor inline outputs/collage
  const lastAssistantId = [...activeConv.messages].filter((m) => m.role === 'assistant').slice(-1)[0]?.id

  const startNewChat = () => {
    const id = `c_${Date.now()}`
    const now = new Date().toISOString()
    setConversations((prev) => [
      { 
        id, 
        title: "New chat", 
        messages: [],
        createdAt: now,
        updatedAt: now
      }, 
      ...prev
    ])
    setActiveId(id)
    // Deduct credits when a new chat is created
    try { decrementCredit(0.25) } catch {}
  }

  const updateActive = (updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === activeId ? updater(c) : c)))
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    const isFirst = (activeConv?.messages?.length || 0) === 0
    const wantsImage = mentionsImage(input)
    // First message: require prompt, and require image, but give a helpful hint if missing
    if (isFirst) {
      if (!input.trim()) {
        updateActive((c) => ({
          ...c,
          messages: [
            ...c.messages,
            { id: `m_${Date.now()}`, role: 'assistant', content: "Please enter a prompt (and attach an image) to start a new ad.", timestamp: new Date().toISOString() },
          ],
        }))
        return
      }
      if (!imageFile) {
        const msg = wantsImage
          ? "It sounds like you're referring to an image (e.g., 'this picture'), but no image is attached. Click the paperclip to attach an image."
          : "Please attach an image to start a new ad. Click the paperclip to upload."
        updateActive((c) => ({
          ...c,
          messages: [
            ...c.messages,
            { id: `m_${Date.now()}`, role: 'assistant', content: msg, timestamp: new Date().toISOString() },
          ],
        }))
        try { fileInputRef.current?.click() } catch {}
        return
      }
    } else {
      // Subsequent messages require at least a prompt
      if (!input.trim()) return
      // If user references an image but there's none in this chat yet, prompt to attach
      if (!imageFile && wantsImage && !hasPreviousImage(activeConv)) {
        updateActive((c) => ({
          ...c,
          messages: [
            ...c.messages,
            { id: `m_${Date.now()}`, role: 'assistant', content: "You mentioned an image, but none is attached in this chat yet. Please attach one with the paperclip.", timestamp: new Date().toISOString() },
          ],
        }))
        try { fileInputRef.current?.click() } catch {}
        return
      }
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
      ...(imageFile && {
        attachments: [{
          type: 'image',
          url: URL.createObjectURL(imageFile)
        }]
      })
    }

    updateActive((c) => ({ ...c, messages: [...c.messages, userMessage] }))

    setBusy(true)
    try {
      const form = new FormData()
      form.append("prompt", input || "")
      if (imageFile) form.append("image", imageFile, imageFile.name || "upload.png")
      // Reuse previous analysis if no new image provided
      if (!imageFile && activeConv?.analysis) {
        try { form.append("analysis", JSON.stringify(activeConv.analysis)) } catch {}
      }
      // Always forward conversation history for better context when composing
      try {
        const history = activeConv.messages.map((m) => ({ role: m.role, content: m.content }))
        form.append("history", JSON.stringify(history))
      } catch {}
      if (platform) form.append("platform", platform)
      if (tone) form.append("tone", tone)
      form.append("num_variants", String(variants || 1))
      try { form.append("sizes", JSON.stringify(sizesForAspect(selectedAspectRatio))) } catch {}

      const res = await fetch("/api/pipeline/run", { method: "POST", body: form })
      if (!res.ok) {
        const t = await res.text().catch(() => "")
        throw new Error(`HTTP ${res.status}: ${t}`)
      }
      const json = (await res.json()) as PipelineResponse

      const textParts: string[] = []
      if (json.copy_best) {
        textParts.push(
          `Headline: ${json.copy_best.headline}\n${json.copy_best.subheadline}\nCTA: ${json.copy_best.cta}`,
        )
      } else if (json.copy_variants?.length) {
        textParts.push(`Generated ${json.copy_variants.length} copy variants.`)
      }
      // capture best copy for front-and-center recommendations panel
      const best = json.copy_best || (json.copy_variants && json.copy_variants[0]) || null
      setRecommendation(best ? { headline: best.headline, subheadline: best.subheadline, cta: best.cta } : null)

      const attachments: Array<{ type: "image"; url: string }> = []
      if (json.thumbnail_url) attachments.push({ type: "image", url: normalizeUrl(json.thumbnail_url) })
      for (const o of json.outputs || []) {
        const fmt = o.format.toLowerCase()
        if (["png", "jpg", "jpeg", "webp", "svg"].includes(fmt)) attachments.push({ type: "image", url: normalizeUrl(o.url) })
      }
      // Keep a normalized, deduped list for the collage
      setLastOutputs(dedupePreferredImages(json.outputs || []))
      setCopyList((json.copy_variants && json.copy_variants.length ? json.copy_variants : (json.copy_best ? [json.copy_best] : [])).map((c: any) => ({ headline: c.headline, subheadline: c.subheadline, cta: c.cta })))

      const assistantMsg: Message = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        // We show a single collage from lastOutputs; avoid duplicating as attachments
        attachments: undefined,
      }

      updateActive((c) => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        analysis: json.analysis ? json.analysis : c.analysis,
      }))

      // Auto-name the conversation after the first successful AI response
      if (isFirst) {
        const title = deriveTitleFromResponse(json, input)
        updateActive((c) => ({ ...c, title }))
      }
    } catch (err: any) {
      updateActive((c) => ({
        ...c,
        messages: [
          ...c.messages,
          { id: `m_${Date.now() + 2}`, role: "assistant", content: `Error: ${err?.message || String(err)}`, timestamp: new Date().toISOString() },
        ],
      }))
    } finally {
      setBusy(false)
      setInput("")
      setImageFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [activeConv.messages.length])

  // Auto-ingest pending data handed off from Home and send once
  const handleSendWithPending = useCallback(async () => {
    if (autoRanRef.current) return
    const hasPending = !!pendingPrompt || (pendingFiles && pendingFiles.length > 0)
    if (!hasPending) return
    autoRanRef.current = true
    
    const firstFile = pendingFiles?.[0] || null
    setInput(pendingPrompt || "")
    setImageFile(firstFile)
    // allow state to commit before sending
    await new Promise((r) => setTimeout(r, 0))
    await handleSendMessage(new Event('submit') as any)
    // clear pending
    setPendingPrompt("")
    setPendingFiles([])
  }, [pendingPrompt, pendingFiles, setPendingPrompt, setPendingFiles, handleSendMessage])

  useEffect(() => {
    handleSendWithPending()
  }, [handleSendWithPending])

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0]
    if (f && f.type.startsWith("image/")) setImageFile(f)
  }

  return (
    <div className="h-full min-h-0 w-full bg-transparent overflow-hidden flex">
      {/* Left: Conversations */}
      <aside className="hidden md:flex w-[400px] lg:w-[480px] flex-col">
        <div className="p-3">
          <div className="relative group isolate">
            <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
            <Button onClick={startNewChat} className="relative z-10 w-full h-16 rounded-2xl border border-zinc-800 bg-black/60 hover:bg-black/70 text-lg transition-all shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40">
              <PlusCircle className="w-5 h-5 mr-2" /> New chat
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto sidebar-scrollbar">
          {conversations.map((c) => (
            <div key={c.id} className="px-3 py-1">
              <div className="relative group isolate">
                <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-5 group-hover:opacity-10 blur-sm transition-opacity z-0"></div>
                <button
                  onClick={() => setActiveId(c.id)}
                  className={`relative z-10 w-full text-left px-4 py-2.5 text-sm rounded-xl border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/30 ${
                    activeId === c.id 
                      ? 'border-indigo-500/40 bg-zinc-900 hover:bg-zinc-900' 
                      : 'border-zinc-800 bg-black/40 hover:bg-black/50'
                  }`}
                  aria-current={activeId === c.id ? 'true' : undefined}
                >
                  {activeId === c.id && (
                    <span className="pointer-events-none absolute left-2 top-2 bottom-2 w-1 rounded-full bg-gradient-to-b from-indigo-500 via-pink-500 to-amber-500 opacity-70" />
                  )}
                  <div className="truncate text-sm font-medium text-zinc-200 pl-1">{c.title || "Untitled"}</div>
                  <div className="text-xs text-zinc-500 pl-1">{c.messages.length} messages</div>
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Vertical separator */}
      <div aria-hidden="true" className="hidden md:block w-px self-stretch bg-gradient-to-b from-transparent via-zinc-800 to-transparent" />

      {/* Right: Chat */}
      <section className="flex-1 min-h-0 flex flex-col">
        {/* Messages & Collage */}
        <div className="flex-1 min-h-0 overflow-y-auto chat-scrollbar p-8 space-y-8 mx-auto w-full max-w-7xl">
          {activeConv.messages.length === 0 && (
            <div className="mx-auto max-w-3xl w-full">
              <div className="mt-2 grid sm:grid-cols-2 gap-4">
                {suggestions.slice(0, 4).map((s, i) => (
                  <div key={i} className="relative group isolate">
                    <div className="pointer-events-none absolute -inset-[2px] rounded-xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
                    <button
                      onClick={() => {
                        if (s.preset) {
                          if (s.preset.platform) setPlatform(s.preset.platform as any)
                          if (s.preset.tone) setTone(s.preset.tone as any)
                          if (s.preset.aspect) setSelectedAspectRatio(s.preset.aspect)
                          if (s.preset.variants) setVariants(s.preset.variants)
                        }
                        applySuggestion(s.text)
                      }}
                      className="relative z-10 w-full text-left rounded-xl border border-zinc-900/60 bg-zinc-900/50 hover:bg-zinc-900/60 p-5"
                    >
                      <div className="text-base text-zinc-200">{s.label}</div>
                      <div className="text-sm text-zinc-400 mt-1 line-clamp-2">{s.text}</div>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Message thread: render both user and assistant messages inline */}
          {activeConv.messages.map((m) => (
            m.role === 'user' ? (
              <div key={m.id} className="flex justify-end">
                <div className="relative group isolate max-w-[85%]">
                  <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
                  <div className="relative z-10 rounded-2xl px-5 py-3.5 border bg-zinc-900/50 border-zinc-900/60">
                    <div className="whitespace-pre-wrap text-zinc-200 text-[15px]">{m.content}</div>
                    {m.attachments && m.attachments.length > 0 && (
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        {m.attachments.map((a, i) => (
                          <a key={i} href={a.url} target="_blank" rel="noreferrer" className="block">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={a.url} alt="attachment" className="rounded-md border border-zinc-900/60 max-h-48 object-contain" />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start">
                <div className="relative group isolate max-w-[85%]">
                  <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-5 group-hover:opacity-10 blur-sm transition-opacity z-0"></div>
                  <div className="relative z-10 rounded-2xl px-5 py-3.5 border bg-zinc-900/40 border-zinc-900/60">
                    {m.content && (
                      <div className="whitespace-pre-wrap text-zinc-200 text-[15px] mb-2">{m.content}</div>
                    )}
                    {/* Recommended copy (if available) */}
                    {recommendation && m.id === lastAssistantId && (
                      <div className="mb-3">
                        <div className="text-sm text-zinc-200">{recommendation.headline}</div>
                        <div className="text-sm text-zinc-400">{recommendation.subheadline}</div>
                        <div className="text-xs text-zinc-500 mt-1">CTA: {recommendation.cta}</div>
                      </div>
                    )}
                    {/* Inline collage anchored to the latest assistant message */}
                    {lastOutputs.length > 0 && m.id === lastAssistantId && (
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {lastOutputs.map((o, i) => (
                          <a key={i} href={o.url} target="_blank" rel="noreferrer" className="relative block group isolate">
                            <span className="pointer-events-none absolute -inset-[2px] rounded-md bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></span>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={o.url} alt={`${o.format} ${o.width}x${o.height}`} className="relative z-10 rounded-md border border-zinc-900/60 bg-zinc-900/40 aspect-square object-contain group-hover:border-zinc-800" />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          ))}

          {/* Inline typing/loading indicator where the assistant response will appear */}
          {busy && (
            <div className="flex justify-start">
              <div className="relative group isolate max-w-[85%]">
                <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-5 blur-sm transition-opacity z-0"></div>
                <div className="relative z-10 rounded-2xl px-5 py-4 border bg-zinc-900/40 border-zinc-900/60">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Anchor for auto-scroll at the end of messages */}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="p-4 bg-transparent shrink-0" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
          <div className="mx-auto w-full max-w-7xl">
          {/* Advanced controls (hidden by default) */}
          {showOptions && (
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <ToggleGroup type="single" value={platform} onValueChange={(v) => setPlatform((v as any) || "")} className="bg-zinc-900/40 border border-zinc-900 rounded-md">
              <ToggleGroupItem value="instagram" aria-label="Instagram" className="px-2 text-xs">IG</ToggleGroupItem>
              <ToggleGroupItem value="facebook" aria-label="Facebook" className="px-2 text-xs">FB</ToggleGroupItem>
              <ToggleGroupItem value="linkedin" aria-label="LinkedIn" className="px-2 text-xs">LI</ToggleGroupItem>
              <ToggleGroupItem value="x" aria-label="X" className="px-2 text-xs">X</ToggleGroupItem>
            </ToggleGroup>
            <ToggleGroup type="single" value={tone} onValueChange={(v) => setTone((v as any) || "")} className="bg-zinc-900/40 border border-zinc-900 rounded-md">
              <ToggleGroupItem value="bold" aria-label="Bold" className="px-2 text-xs">B</ToggleGroupItem>
              <ToggleGroupItem value="friendly" aria-label="Friendly" className="px-2 text-xs">F</ToggleGroupItem>
              <ToggleGroupItem value="professional" aria-label="Professional" className="px-2 text-xs">P</ToggleGroupItem>
              <ToggleGroupItem value="minimal" aria-label="Minimal" className="px-2 text-xs">M</ToggleGroupItem>
            </ToggleGroup>
            <Select value={String(variants)} onValueChange={(v) => setVariants(Number(v || 3))}>
              <SelectTrigger className="bg-zinc-900/60 hover:bg-zinc-900 border border-zinc-900 rounded-md h-8 px-2 text-xs">
                <SelectValue placeholder="Variants" />
              </SelectTrigger>
              <SelectContent className="bg-zinc-950 border border-zinc-900 text-zinc-200">
                <SelectItem value="1" className="text-zinc-200 focus:bg-zinc-900">1 variant</SelectItem>
                <SelectItem value="3" className="text-zinc-200 focus:bg-zinc-900">3 variants (recommended)</SelectItem>
                <SelectItem value="5" className="text-zinc-200 focus:bg-zinc-900">5 variants</SelectItem>
              </SelectContent>
            </Select>
          </div>
          )}

          {/* AI suggestions removed */}

          <form className="flex items-end gap-3" onSubmit={handleSendMessage}>
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={(e) => setImageFile(e.target.files?.[0] || null)}
              className="hidden"
            />
            <div className="flex-1">
              <div className="relative group isolate">
                {/* Chromatic glow background (behind) */}
                <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-30 blur-md z-0" />
                <div className="relative z-10 rounded-2xl border border-zinc-800 bg-black/50">
                  {/* Upload (left) */}
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute left-2 bottom-2 w-12 h-12 rounded-full bg-zinc-900/70 hover:bg-zinc-900 text-zinc-200 flex items-center justify-center border border-zinc-700/60 shadow-[0_0_20px_rgba(99,102,241,0.25)]"
                    title="Attach image"
                  >
                    <Paperclip className="w-5 h-5" />
                  </button>
                  {/* Textarea */}
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault()
                        handleSendMessage(new Event('submit') as any)
                      }
                    }}
                    placeholder={activeConv.messages.length === 0 ? "Attach an image and enter a prompt to start..." : "Type your prompt (Shift+Enter for newline)..."}
                    className="flex-1 bg-black/40 border-none min-h-[100px] pl-16 pr-48 py-4 text-base resize-none"
                    rows={3}
                  />
                  {/* Right controls */}
                  <div className="absolute right-2 bottom-2 flex items-center gap-2">
                    {/* Preferences */}
                    <Popover>
                      <PopoverTrigger asChild>
                        <button
                          type="button"
                          className="relative w-12 h-12 rounded-full bg-zinc-900/70 hover:bg-zinc-900 text-zinc-200 border border-zinc-700/60 flex items-center justify-center shadow-[0_0_22px_rgba(236,72,153,0.25)]"
                          title="Preferences"
                        >
                          <SlidersHorizontal className="w-5 h-5" />
                        </button>
                      </PopoverTrigger>
                      <PopoverContent align="end" className="w-80 bg-zinc-950 border-zinc-900 text-zinc-200">
                        <div className="space-y-3">
                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Platform</div>
                            <ToggleGroup type="single" value={platform} onValueChange={(v) => setPlatform((v as any) || "") } className="bg-zinc-900/40 border border-zinc-900 rounded-md">
                              <ToggleGroupItem value="instagram" aria-label="Instagram" className="px-2 text-xs">IG</ToggleGroupItem>
                              <ToggleGroupItem value="facebook" aria-label="Facebook" className="px-2 text-xs">FB</ToggleGroupItem>
                              <ToggleGroupItem value="linkedin" aria-label="LinkedIn" className="px-2 text-xs">LI</ToggleGroupItem>
                              <ToggleGroupItem value="x" aria-label="X" className="px-2 text-xs">X</ToggleGroupItem>
                            </ToggleGroup>
                          </div>
                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Tone</div>
                            <ToggleGroup type="single" value={tone} onValueChange={(v) => setTone((v as any) || "") } className="bg-zinc-900/40 border border-zinc-900 rounded-md">
                              <ToggleGroupItem value="bold" aria-label="Bold" className="px-2 text-xs">B</ToggleGroupItem>
                              <ToggleGroupItem value="friendly" aria-label="Friendly" className="px-2 text-xs">F</ToggleGroupItem>
                              <ToggleGroupItem value="professional" aria-label="Professional" className="px-2 text-xs">P</ToggleGroupItem>
                              <ToggleGroupItem value="minimal" aria-label="Minimal" className="px-2 text-xs">M</ToggleGroupItem>
                            </ToggleGroup>
                          </div>
                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Variants</div>
                            <Select value={String(variants)} onValueChange={(v) => setVariants(Number(v || 3))}>
                              <SelectTrigger className="bg-zinc-900/60 hover:bg-zinc-900 border border-zinc-900 rounded-md h-8 px-2">
                                <SelectValue placeholder="Select variants" />
                              </SelectTrigger>
                              <SelectContent className="bg-zinc-950 border border-zinc-900 text-zinc-200">
                                <SelectItem value="1" className="text-zinc-200 focus:bg-zinc-900">1 variant</SelectItem>
                                <SelectItem value="3" className="text-zinc-200 focus:bg-zinc-900">3 variants (recommended)</SelectItem>
                                <SelectItem value="5" className="text-zinc-200 focus:bg-zinc-900">5 variants</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <div className="text-xs text-zinc-400 mb-1">Dimensions</div>
                            <Select value={selectedAspectRatio} onValueChange={(v) => setSelectedAspectRatio(v)}>
                              <SelectTrigger className="bg-zinc-900/60 hover:bg-zinc-900 border border-zinc-900 rounded-md h-8 px-2">
                                <SelectValue placeholder="Dimensions" />
                              </SelectTrigger>
                              <SelectContent className="bg-zinc-950 border border-zinc-900 text-zinc-200">
                                {dimensionOptions.map((o) => (
                                  <SelectItem key={o.value} value={o.value} className="text-zinc-200 focus:bg-zinc-900">
                                    {o.label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      </PopoverContent>
                    </Popover>
                    {/* Send */}
                    <button
                      type="submit"
                      disabled={busy || !input.trim()}
                      className="relative w-12 h-12 rounded-full flex items-center justify-center text-white disabled:opacity-60"
                      title="Send"
                    >
                      <span className="absolute -inset-[2px] rounded-full bg-[conic-gradient(at_0%_0%,#f59e0b_0deg,#6366f1_120deg,#ec4899_240deg,#f59e0b_360deg)] opacity-30 blur-md z-0" />
                      {busy ? (
                        <div className="relative z-10 w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      ) : (
                        <div className="relative z-10 w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-700 flex items-center justify-center">
                          <Send className="w-5 h-5" />
                        </div>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {imageFile && (
                <div className="mt-2 flex items-center gap-2 text-sm text-zinc-400">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={URL.createObjectURL(imageFile)} alt="preview" className="w-12 h-12 rounded object-cover border border-zinc-900/60" />
                  <span><ImageIcon className="w-5 h-5 inline mr-1" /> {imageFile.name}</span>
                  <button
                    type="button"
                    className="text-zinc-300 underline"
                    onClick={() => {
                      setImageFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}
              {!imageFile && activeConv?.analysis && (
                <div className="mt-2 text-xs text-zinc-400">Reusing previously attached image for this chat.</div>
              )}
            </div>
          </form>
          </div>
        </div>
      </section>
    </div>
  )
}
