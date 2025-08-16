"use client"

import React, { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ImageIcon, Send, PlusCircle, Paperclip, Square, RectangleVertical, RectangleHorizontal } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useApp } from "@/components/app-context"

// Types matching the API route logs
// Minimal chat message + conversation types
type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  attachments?: Array<{ type: "image"; url: string; variant?: string }>
  timestamp: string
}

type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
}

type PipelineResponse = {
  ok: boolean
  job_id: string
  copy_best?: { headline: string; subheadline: string; cta: string } | null
  copy_variants?: Array<{ headline: string; subheadline: string; cta: string }>
  composition?: { composition_id: string; svg: string; layout_data: any }
  render?: { outputs: Array<{ format: string; width: number; height: number; url: string; variant?: string }>; thumbnail_url?: string }
  qa?: any
  export?: any
  logs: any[]
  thumbnail_url?: string
  outputs?: Array<{ format: string; width: number; height: number; url: string; variant?: string }>
}

// Normalize pipeline asset URLs to use the exposed localhost port in dev
const PIPELINE_BASE = process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8010"
function normalizeUrl(u: string): string {
  try {
    // Relative paths from backend (e.g. /static/..., /outputs/...)
    if (
      u.startsWith("/static/") ||
      u.startsWith("static/") ||
      u.startsWith("/outputs/") ||
      u.startsWith("outputs/")
    ) {
      const base = (PIPELINE_BASE || "http://localhost:8010").replace(/\/$/, "")
      const path = u.startsWith("/") ? u : `/${u}`
      return `${base}${path}`
    }

    const url = new URL(u)
    if (url.port === "8000") {
      // Map container internal port -> host mapped port
      url.port = "8010"
      if (url.hostname === "0.0.0.0") url.hostname = "localhost"
      return url.toString()
    }
    return u
  } catch {
    return u
  }
}

export function ChatView() {
  const [prompt, setPrompt] = useState("")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const autoRanRef = useRef(false)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  // Pending handoff from HomeView
  const { pendingPrompt, setPendingPrompt, pendingFiles, setPendingFiles } = useApp()

  // Platform selector state
  const [selectedVariants, setSelectedVariants] = useState<string[]>(["square", "portrait"]) // defaults
  const variantOptions: Record<string, { label: string; width: number; height: number }> = {
    square: { label: "Instagram (Square)", width: 1080, height: 1080 },
    portrait: { label: "Instagram", width: 1080, height: 1350 },
    landscape: { label: "YouTube", width: 1920, height: 1080 },
    story: { label: "Instagram Stories", width: 1080, height: 1920 },
  }
  const variantShortLabel: Record<string, string> = {
    square: "Square",
    portrait: "Instagram",
    landscape: "YouTube",
    story: "Story",
  }

  const [conversations, setConversations] = useState<Conversation[]>([
    { id: "c_default", title: "New chat", messages: [] },
  ])
  const [activeId, setActiveId] = useState<string>("c_default")

  const activeConv = useMemo(() => conversations.find((c) => c.id === activeId)!, [conversations, activeId])

  const startNewChat = () => {
    const id = `c_${Date.now()}`
    setConversations((prev) => [{ id, title: "New chat", messages: [] }, ...prev])
    setActiveId(id)
  }

  const updateActive = (updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === activeId ? updater(c) : c)))
  }

  async function handleSend(e?: React.FormEvent) {
    e?.preventDefault()
    if (!prompt && !imageFile) return

    const userMsg: ChatMessage = {
      id: `m_${Date.now()}`,
      role: "user",
      content: prompt,
      timestamp: new Date().toISOString(),
      attachments: imageFile ? [{ type: "image", url: URL.createObjectURL(imageFile) }] : undefined,
    }

    updateActive((c) => ({
      ...c,
      title: c.messages.length === 0 && prompt ? prompt.slice(0, 40) : c.title,
      messages: [...c.messages, userMsg],
    }))

    setBusy(true)
    try {
      const form = new FormData()
      form.append("prompt", prompt || "")
      if (imageFile) form.append("image", imageFile, imageFile.name || "upload.png")

      // Include selected sizes for the orchestrator
      const sizes = selectedVariants.map((v) => ({
        name: v,
        width: variantOptions[v].width,
        height: variantOptions[v].height,
      }))
      if (sizes.length > 0) form.append("sizes", JSON.stringify(sizes))

      // Include full past context for this chat (role + content only)
      const historyPayload = [...activeConv.messages, userMsg].map((m) => ({ role: m.role, content: m.content }))
      form.append("history", JSON.stringify(historyPayload))

      const res = await fetch("/api/pipeline/run", { method: "POST", body: form })
      if (!res.ok) {
        const t = await res.text().catch(() => "")
        throw new Error(`HTTP ${res.status}: ${t}`)
      }
      const json = (await res.json()) as PipelineResponse

      // We no longer include textual copy in chat output – images only

      // Select exactly one JPG per variant (aspect ratio) and order them
      const order = ["square", "portrait", "landscape", "story"]
      const byVariant = new Map<string, string>()
      for (const o of json.outputs || []) {
        const fmt = (o.format || "").toLowerCase()
        if (fmt === "jpg" || fmt === "jpeg") {
          const v = (o.variant as string) || "default"
          if (!byVariant.has(v)) byVariant.set(v, normalizeUrl(o.url))
        }
      }
      const attachments: Array<{ type: "image"; url: string; variant?: string }> = []
      for (const v of order) {
        const url = byVariant.get(v)
        if (url) attachments.push({ type: "image", url, variant: v })
      }
      // Fallback: if no known variants, include whatever jpgs exist
      if (attachments.length === 0) {
        for (const [_, url] of byVariant) attachments.push({ type: "image", url })
      }

      const assistantMsg: ChatMessage = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        attachments: attachments.length ? attachments : undefined,
      }

      updateActive((c) => ({ ...c, messages: [...c.messages, assistantMsg] }))
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
      setPrompt("")
      setImageFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  // Auto-ingest pending data handed off from Home and send once
  useEffect(() => {
    if (autoRanRef.current) return
    const hasPending = !!pendingPrompt || (pendingFiles && pendingFiles.length > 0)
    if (!hasPending) return
    autoRanRef.current = true
    ;(async () => {
      const firstFile = pendingFiles?.[0] || null
      setPrompt(pendingPrompt || "")
      setImageFile(firstFile)
      // allow state to commit before sending
      await new Promise((r) => setTimeout(r, 0))
      await handleSend()
      // clear pending
      setPendingPrompt("")
      setPendingFiles([])
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPrompt, pendingFiles])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [activeConv.messages.length])

  return (
    <div className="h-[75vh] min-h-0 bg-zinc-950/60 border border-zinc-900 rounded-2xl overflow-hidden flex">
      {/* Left: Conversations */}
      <aside className="hidden md:flex w-64 lg:w-72 border-r border-zinc-900 flex-col min-h-0">
        <div className="p-3">
          <Button onClick={startNewChat} className="w-full rounded-lg">
            <PlusCircle className="w-4 h-4 mr-2" /> New chat
          </Button>
        </div>
        <ScrollArea className="flex-1 min-h-0">
          <div>
            {conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => setActiveId(c.id)}
                className={`w-full text-left px-3 py-2 hover:bg-zinc-900/60 ${c.id === activeId ? "bg-zinc-900/60" : ""}`}
              >
                <div className="truncate text-sm text-zinc-200">{c.title || "Untitled"}</div>
                <div className="text-xs text-zinc-500">{c.messages.length} messages</div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </aside>

      {/* Right: Chat */}
      <section className="flex-1 flex flex-col min-h-0">
        {/* Messages */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-4 space-y-4">
            {activeConv.messages.length === 0 && (
              <div className="text-center text-zinc-500 mt-10">Start by typing a prompt or attaching an image.</div>
            )}
            {activeConv.messages.map((m) => {
              const isUser = m.role === "user"
              const hasAttachments = !!(m.attachments && m.attachments.length > 0)
              return (
                <div key={m.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                  {isUser ? (
                    <div className="max-w-[80%] rounded-2xl px-4 py-3 border bg-zinc-900/60 border-zinc-800">
                      {m.content?.trim() ? (
                        <div className="whitespace-pre-wrap text-zinc-200">{m.content}</div>
                      ) : null}
                      {hasAttachments && (
                        <div className="mt-2 grid grid-cols-2 gap-2">
                          {m.attachments!.map((a, i) => (
                            <a key={i} href={a.url} target="_blank" rel="noreferrer" className="block">
                              <img src={a.url} alt="attachment" className="rounded-md border border-zinc-800 max-h-48 object-contain" />
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : hasAttachments ? (
                    <div className="w-full max-w-3xl">
                      {(() => {
                        const n = m.attachments!.length
                        if (n === 1) {
                          const a = m.attachments![0]
                          return (
                            <a href={a.url} target="_blank" rel="noreferrer" className="block group">
                              <img src={a.url} alt="output" className="w-full max-h-[480px] object-contain rounded-2xl border border-zinc-800/70 bg-zinc-900/30 group-hover:bg-zinc-900/40 transition" />
                            </a>
                          )
                        }
                        if (n === 2) {
                          return (
                            <div className="grid grid-cols-2 gap-2">
                              {m.attachments!.map((a, i) => (
                                <a key={i} href={a.url} target="_blank" rel="noreferrer" className="block group">
                                  <img src={a.url} alt="output" className="w-full h-64 object-cover rounded-2xl border border-zinc-800/70 bg-zinc-900/30 group-hover:bg-zinc-900/40 transition" />
                                </a>
                              ))}
                            </div>
                          )
                        }
                        if (n === 3) {
                          return (
                            <div className="grid grid-cols-3 auto-rows-[140px] md:auto-rows-[180px] gap-2">
                              {m.attachments!.map((a, i) => (
                                <a
                                  key={i}
                                  href={a.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className={`block group ${i === 0 ? "col-span-2 row-span-2" : ""}`}
                                >
                                  <img src={a.url} alt="output" className="w-full h-full object-cover rounded-2xl border border-zinc-800/70 bg-zinc-900/30 group-hover:bg-zinc-900/40 transition" />
                                </a>
                              ))}
                            </div>
                          )
                        }
                        // 4 or more
                        return (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 auto-rows-[120px] md:auto-rows-[160px]">
                            {m.attachments!.slice(0, 8).map((a, i) => (
                              <a key={i} href={a.url} target="_blank" rel="noreferrer" className="block group">
                                <img src={a.url} alt="output" className="w-full h-full object-cover rounded-2xl border border-zinc-800/70 bg-zinc-900/30 group-hover:bg-zinc-900/40 transition" />
                              </a>
                            ))}
                          </div>
                        )
                      })()}
                    </div>
                  ) : (
                    <div className="max-w-[80%] rounded-2xl px-4 py-3 border bg-zinc-900/40 border-zinc-800">
                      {m.content?.trim() ? (
                        <div className="whitespace-pre-wrap text-zinc-200">{m.content}</div>
                      ) : null}
                    </div>
                  )}
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Composer */}
        <div className="border-t border-zinc-900 p-3">
          {/* Platform selector (intuitive cards) */}
          <div className="mb-2">
            <div className="text-xs text-zinc-400 mb-1">Select platforms</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {(
                [
                  { key: "square", icon: Square, label: variantOptions.square.label, hint: "1:1" },
                  { key: "portrait", icon: RectangleVertical, label: variantOptions.portrait.label, hint: "4:5" },
                  { key: "landscape", icon: RectangleHorizontal, label: variantOptions.landscape.label, hint: "16:9" },
                  { key: "story", icon: RectangleVertical, label: variantOptions.story.label, hint: "9:16" },
                ] as const
              ).map((opt) => {
                const selected = selectedVariants.includes(opt.key)
                const Icon = opt.icon
                return (
                  <button
                    key={opt.key}
                    type="button"
                    title={opt.label}
                    onClick={() =>
                      setSelectedVariants((prev) =>
                        prev.includes(opt.key) ? prev.filter((k) => k !== opt.key) : [...prev, opt.key]
                      )
                    }
                    className={`text-left rounded-xl border px-3 py-2 transition ${
                      selected ? "border-indigo-500/50 bg-indigo-500/10" : "border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4 text-zinc-300" />
                      <div className="text-xs text-zinc-300 truncate">{opt.label}</div>
                    </div>
                    <div className="text-[10px] text-zinc-500 mt-0.5">{opt.hint}</div>
                  </button>
                )
              })}
            </div>
          </div>
          <form className="flex items-center gap-2" onSubmit={handleSend}>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={(e) => setImageFile(e.target.files?.[0] || null)}
            />
            <Button type="button" variant="secondary" onClick={() => fileInputRef.current?.click()} disabled={busy}>
              <Paperclip className="w-4 h-4 mr-2" /> Attach
            </Button>
            <Input
              className="flex-1"
              placeholder="Send a message..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={busy}
            />
            <Button type="submit" disabled={busy || (!prompt && !imageFile)}>
              {busy ? "Sending..." : (<><Send className="w-4 h-4 mr-2" /> Send</>)}
            </Button>
          </form>
          {imageFile && (
            <div className="text-xs text-zinc-400 mt-2 flex items-center gap-2">
              <ImageIcon className="w-4 h-4" /> Attached: {imageFile.name}
              <button
                className="text-zinc-300 underline"
                onClick={() => {
                  setImageFile(null)
                  if (fileInputRef.current) fileInputRef.current.value = ""
                }}
              >
                Remove
              </button>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

