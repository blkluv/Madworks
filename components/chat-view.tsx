"use client"

import React, { useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ImageIcon, Send, PlusCircle, Paperclip } from "lucide-react"

// Types matching the API route logs
// Minimal chat message + conversation types
type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  attachments?: Array<{ type: "image"; url: string }>
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
  render?: { outputs: Array<{ format: string; width: number; height: number; url: string }>; thumbnail_url?: string }
  qa?: any
  export?: any
  logs: any[]
  thumbnail_url?: string
  outputs?: Array<{ format: string; width: number; height: number; url: string }>
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

      const attachments: Array<{ type: "image"; url: string }> = []
      if (json.thumbnail_url) attachments.push({ type: "image", url: normalizeUrl(json.thumbnail_url) })
      for (const o of json.outputs || []) {
        const fmt = o.format.toLowerCase()
        if (["png", "jpg", "jpeg", "webp", "svg"].includes(fmt)) attachments.push({ type: "image", url: normalizeUrl(o.url) })
      }

      const assistantMsg: ChatMessage = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: textParts.join("\n\n") || "Generated.",
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

  return (
    <div className="h-[75vh] bg-zinc-950/60 border border-zinc-900 rounded-2xl overflow-hidden flex">
      {/* Left: Conversations */}
      <aside className="hidden md:flex w-64 lg:w-72 border-r border-zinc-900 flex-col">
        <div className="p-3">
          <Button onClick={startNewChat} className="w-full rounded-lg">
            <PlusCircle className="w-4 h-4 mr-2" /> New chat
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto">
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
      </aside>

      {/* Right: Chat */}
      <section className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeConv.messages.length === 0 && (
            <div className="text-center text-zinc-500 mt-10">Start by typing a prompt or attaching an image.</div>
          )}
          {activeConv.messages.map((m) => (
            <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 border ${m.role === "user" ? "bg-zinc-900/60 border-zinc-800" : "bg-black/60 border-zinc-900"}`}>
                <div className="whitespace-pre-wrap text-zinc-200">{m.content}</div>
                {m.attachments && m.attachments.length > 0 && (
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    {m.attachments.map((a, i) => (
                      <a key={i} href={a.url} target="_blank" rel="noreferrer" className="block">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={a.url} alt="attachment" className="rounded-md border border-zinc-800 max-h-48 object-contain" />
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Composer */}
        <div className="border-t border-zinc-900 p-3">
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
