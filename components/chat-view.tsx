"use client"

import { useState, useRef, useEffect, useCallback } from 'react'
import { useSession } from "next-auth/react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { PlusCircle, Paperclip, Image as ImageIcon, Send, SlidersHorizontal } from "lucide-react"
import { useApp } from "@/components/app-context"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"
import { CreditsPill } from "@/components/credits-pill"

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  attachments?: Array<{ type: string; url: string; href?: string; variant?: string; label?: string; size?: string }>
}

// Robust unique ID generator to avoid React key collisions when sending quickly
const uid = () => {
  try {
    // @ts-ignore
    if (typeof crypto !== 'undefined' && crypto?.randomUUID) return crypto.randomUUID()
  } catch {}
  return `id_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

// Convert an SVG URL into a data URL and inline any external raster references
// (href/xlink:href) as data URIs so the preview always renders in <img>.
async function toDisplayUrl(o: { url: string; format?: string }): Promise<string> {
  try {
    let u = o?.url || ""
    const fmt = (o?.format || "").toLowerCase()
    if (!u) return u
    if (u.startsWith("data:")) return u
    const isSvg = fmt === "svg" || u.toLowerCase().endsWith(".svg")
    if (!isSvg) return u

    // Fetch the SVG text (direct, no proxy) with cache busting
    const svgUrl = finalizeDirect(u)
    const res = await fetch(svgUrl, { cache: "no-store" })
    if (!res.ok) return svgUrl
    let svgText = await res.text()

    // Find all href/xlink:href references and inline them as data URLs
    const hrefRe = /(xlink:href|href)\s*=\s*(["'])(.*?)\2/gi
    const found: Array<{ full: string; attr: string; quote: string; src: string }> = []
    svgText.replace(hrefRe, (_m, attr, quote, src) => {
      if (src && !src.startsWith("data:")) {
        found.push({ full: _m, attr, quote, src })
      }
      return _m
    })

    // Helper: blob -> data URL
    const blobToDataUrl = (blob: Blob) => new Promise<string>((resolve, reject) => {
      try {
        const fr = new FileReader()
        fr.onload = () => resolve(String(fr.result || ""))
        fr.onerror = (e) => reject(e)
        fr.readAsDataURL(blob)
      } catch (e) { reject(e) }
    })

    // Minimal HTML entity decode (avoid bringing an extra dependency)
    const decodeEntities = (s: string) =>
      s
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")

    for (const f of found) {
      try {
        // Normalize to an absolute direct URL (no proxy) for fetching.
        // Decode HTML entities like &amp; to avoid requests like ...jpg%26t%3D...
        const raw = decodeEntities(f.src)
        const abs = finalizeDirect(normalizeUrl(raw))
        const r = await fetch(abs, { cache: "no-store" })
        if (!r.ok) continue
        const blob = await r.blob()
        const dataUrl = await blobToDataUrl(blob)
        // Replace only this attribute occurrence to avoid accidental over-replace
        const safe = f.full.replace(f.src, dataUrl)
        svgText = svgText.replace(f.full, safe)
      } catch {}
    }

    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`
  } catch (err) {
    console.error("toDisplayUrl error", err, { url: o?.url, format: o?.format })
    return o?.url
  }
}

// Prefer a single best format per size/variant to avoid duplicates in the collage
function dedupePreferredImages(
  outs: Array<{ format: string; width: number; height: number; url: string; variant?: string; label?: string; size?: string }> = []
) {
  // Prefer raster formats when available, but allow SVG as a fallback so users still see outputs
  const rank: Record<string, number> = { png: 4, jpg: 3, jpeg: 3, webp: 2, svg: 1 }
  const bestByKey = new Map<string, { format: string; width: number; height: number; url: string; variant?: string }>()
  for (const o of outs) {
    const fmt = (o.format || "").toLowerCase()
    const key = `${o.width}x${o.height}_${o.variant || ""}`
    const existing = bestByKey.get(key)
    if (!existing || (rank[fmt] || 0) > (rank[(existing.format || "").toLowerCase()] || 0)) {
      bestByKey.set(key, o)
    }
  }
  // Do not normalize here; finalize per-attachment later to apply proxy and cache-busting appropriately
  return Array.from(bestByKey.values())
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
  composition?: any
  logs?: Array<{ step: string; status: string; message?: string; error?: string }>
}

function normalizeUrl(u: string): string {
  if (!u) return u
  try {
    let v = u.trim()
    // Attempt to decode percent-encoded URLs like http%3A//...
    if (/%[0-9A-Fa-f]{2}/.test(v)) {
      try { v = decodeURIComponent(v) } catch {}
    }
    // If already absolute or data/blob URL after decoding, return as-is
    if (v.startsWith("http://") || v.startsWith("https://") || v.startsWith("data:") || v.startsWith("blob:")) return v

    // Do NOT remap app-local routes. Keep them relative to the app origin.
    if (v.startsWith("/api/") || v.startsWith("/_next/") || v.startsWith("/favicon") || v === "/") return v

    // Otherwise prefix with pipeline base
    const base = (process.env.NEXT_PUBLIC_PIPELINE_BASE || process.env.NEXT_PUBLIC_PIPELINE_URL || "http://localhost:8010").replace(/\/$/, "")
    const path = v.startsWith("/") ? v : `/${v}`
    const abs = `${base}${path}`
    console.debug("normalizeUrl", { input: u, decoded: v, base, abs })
    return abs
  } catch {
    return u
  }
}

// Append a cache-busting param to force refresh in the browser
function addCacheBust(u: string): string {
  try {
    if (!u || u.startsWith('data:')) return u
    const hasQ = u.includes('?')
    const sep = hasQ ? '&' : '?'
    return `${u}${sep}cb=${Date.now()}`
  } catch {
    return u
  }
}

// Detect if we need to proxy to avoid mixed-content (https page loading http image)
function needsProxy(u: string): boolean {
  try {
    if (typeof window === 'undefined') return false
    const loc = window.location
    const target = new URL(u, loc.href)
    // Only proxy to avoid mixed-content (https page loading http image). Cross-origin alone is fine.
    const protocolMismatch = loc.protocol === 'https:' && target.protocol === 'http:'
    return protocolMismatch
  } catch { return false }
}

function toProxy(u: string): string {
  return `/api/proxy?u=${encodeURIComponent(u)}&t=${Date.now()}`
}

function finalizeUrl(u: string): string {
  try {
    const abs = normalizeUrl(u)
    if (!abs || abs.startsWith('data:')) return abs
    return needsProxy(abs) ? toProxy(abs) : addCacheBust(abs)
  } catch {
    return u
  }
}

function canonical(u: string): string {
  try {
    // Strip proxy wrapper and cache-busting for equality checks
    let raw = u
    if (raw.startsWith('/api/proxy')) {
      const qs = raw.split('?')[1] || ''
      const sp = new URLSearchParams(qs)
      raw = sp.get('u') || raw
    }
    const url = new URL(raw)
    url.searchParams.delete('cb')
    url.searchParams.delete('t')
    return `${url.origin}${url.pathname}`
  } catch { return u }
}

// When opening in a new tab, prefer the original (de-proxied) URL so the browser renders it natively.
function deproxyUrl(u: string): string {
  try {
    if (!u) return u
    if (!u.startsWith('/api/proxy')) return u
    const sp = new URL(u, typeof window !== 'undefined' ? window.location.href : 'http://localhost').searchParams
    return sp.get('u') || u
  } catch { return u }
}

// Finalize URL without using the proxy (useful for <img> src to avoid proxy edge-cases)
function finalizeDirect(u: string): string {
  try {
    const abs = normalizeUrl(u)
    if (!abs || abs.startsWith('data:')) return abs
    return addCacheBust(abs)
  } catch { return u }
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
  // Removed variants selection (always generate 3 fixed presets server-side)
  const [textColorOverride, setTextColorOverride] = useState<string>("")
  const [recommendation, setRecommendation] = useState<{
    headline: string
    subheadline: string
    cta: string
  } | null>(null)
  const [copyList, setCopyList] = useState<Array<{ headline: string; subheadline: string; cta: string }>>([])
  const [showOptions, setShowOptions] = useState(false)
  const { data: session } = useSession()

  // Shuffle helper for suggestions
  function shuffle<T>(arr: T[]): T[] {
    const a = [...arr]
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a
  }

  const suggestions: Array<{ label: string; text: string; preset?: { platform?: string; tone?: string; aspect?: string } }> = [
    {
      label: "Stories variants",
      text: "Try a few bold, high-contrast Story options. Keep headlines punchy and legible.",
      preset: { platform: "instagram", tone: "bold", aspect: "9:16" },
    },
    {
      label: "Credible LinkedIn",
      text: "Professional, concise tone. Suggest a clean layout and clear CTA.",
      preset: { platform: "linkedin", tone: "professional", aspect: "1:1" },
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
    {
      label: "Bold contrast",
      text: "High contrast type, large CTA. Emphasize readability over flourish.",
      preset: { platform: "instagram", tone: "bold", aspect: "1:1" },
    },
    {
      label: "Minimalist",
      text: "Sparse copy, plenty of negative space, one strong accent color.",
      preset: { tone: "minimal", aspect: "4:5" },
    },
  ]

  // Randomize which four suggestions show
  const [suggestions4, setSuggestions4] = useState(suggestions.slice(0, 4))
  useEffect(() => {
    setSuggestions4(shuffle(suggestions).slice(0, 4))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
    const id = `c_${uid()}`
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
            { id: uid(), role: 'assistant', content: "Please enter a prompt (and attach an image) to start a new ad.", timestamp: new Date().toISOString() },
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
            { id: uid(), role: 'assistant', content: msg, timestamp: new Date().toISOString() },
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
            { id: uid(), role: 'assistant', content: "You mentioned an image, but none is attached in this chat yet. Please attach one with the paperclip.", timestamp: new Date().toISOString() },
          ],
        }))
        try { fileInputRef.current?.click() } catch {}
        return
      }
    }

    const userMessage: Message = {
      id: uid(),
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
      // If an image is attached, upload it directly to the pipeline to avoid large payloads through Next.js
      let analysisForSend: any = null
      if (imageFile) {
        const ingestForm = new FormData()
        ingestForm.append("image", imageFile, imageFile.name || "upload.png")
        const pipelineUrl = (process.env.NEXT_PUBLIC_PIPELINE_URL || process.env.NEXT_PUBLIC_PIPELINE_BASE || "http://localhost:8010").replace(/\/$/, "")
        const ingestRes = await fetch(`${pipelineUrl}/ingest-analyze`, { method: "POST", body: ingestForm })
        if (!ingestRes.ok) {
          const t = await ingestRes.text().catch(() => "")
          throw new Error(`Ingest failed HTTP ${ingestRes.status}: ${t}`)
        }
        analysisForSend = await ingestRes.json()
        // Persist analysis to the conversation for reuse in follow-ups
        updateActive((c) => ({ ...c, analysis: analysisForSend }))
      } else if (activeConv?.analysis) {
        analysisForSend = activeConv.analysis
      }

      const form = new FormData()
      form.append("prompt", input || "")
      if (analysisForSend) {
        try { form.append("analysis", JSON.stringify(analysisForSend)) } catch {}
      }
      // Always forward conversation history for better context when composing
      try {
        const history = activeConv.messages.map((m) => ({ role: m.role, content: m.content }))
        form.append("history", JSON.stringify(history))
      } catch {}
      if (platform) form.append("platform", platform)
      if (tone) form.append("tone", tone)
      try { form.append("sizes", JSON.stringify(sizesForAspect(selectedAspectRatio))) } catch {}
      if (textColorOverride && textColorOverride.trim()) form.append("text_color_override", textColorOverride.trim())

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
      const res = await fetch(`${apiUrl}/api/pipeline/run`, { method: "POST", body: form })
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
        const v0 = json.copy_variants[0]
        textParts.push(
          v0 ? `Headline: ${v0.headline}\n${v0.subheadline}\nCTA: ${v0.cta}` : `Generated ${json.copy_variants.length} copy variants.`
        )
      }

      // If pipeline reported errors and we have no outputs, append brief notes to help the user
      const hadNoOutputs = !(json.outputs && json.outputs.length > 0)
      if (hadNoOutputs && Array.isArray(json.logs)) {
        const errs = json.logs.filter((l: any) => l && l.status === 'error').slice(0, 2)
        if (errs.length) {
          const notes = errs.map((l: any) => `${l.step}: ${l.error || l.message || 'error'}`).join('\n')
          textParts.push(`Notes:\n${notes}`)
        }
      }

      const bestForList = json.copy_best || (json.copy_variants && json.copy_variants[0]) || null
      setCopyList((json.copy_variants && json.copy_variants.length ? json.copy_variants : (json.copy_best ? [json.copy_best] : [])).map((c: any) => ({ headline: c.headline, subheadline: c.subheadline, cta: c.cta })))

      // Build deduped image attachments per message to preserve history
      const deduped = dedupePreferredImages(json.outputs || [])
      if (!deduped.length) {
        try { console.warn("pipeline returned no outputs", { thumbnail: json.thumbnail_url, logs: json.logs }) } catch {}
      }
      // Ensure SVGs are embedded as data URLs so they always render in <img>; apply proxy/bust for others
      const attProcessed = await Promise.all(
        deduped.map(async (o: any) => {
          const fmt = (o?.format || '').toLowerCase()
          const isSvg = fmt === 'svg' || (o?.url || '').toLowerCase().endsWith('.svg')
          const baseUrl = normalizeUrl(o?.url || '')
          // For image element, embed SVGs to avoid content-type/CORS quirks; for raster, prefer direct URL (no proxy) to avoid early 404s, with cache-bust.
          const viewUrl = isSvg ? await toDisplayUrl({ url: baseUrl, format: 'svg' }) : finalizeDirect(baseUrl)
          const href = finalizeUrl(baseUrl)
          return {
            type: 'image' as const,
            url: viewUrl,
            href,
            variant: [o.variant, o.size].filter(Boolean).join(' ') || undefined,
          }
        })
      )
      let attachments: Array<{ type: "image"; url: string; href?: string; variant?: string }> = attProcessed as any

      // If a thumbnail_url is provided, use it as the first preview attachment (normalized and SVG-embedded if needed)
      try {
        const thumb = (json as any)?.thumbnail_url as string | undefined
        if (thumb) {
          const normalized = normalizeUrl(thumb)
          const previewImgUrl = /\.svg(\?.*)?$/i.test(normalized)
            ? await toDisplayUrl({ url: normalized, format: 'svg' })
            : finalizeUrl(normalized)
          const previewHref = finalizeUrl(normalized)
          // Avoid duplicates if the same URL already exists in attachments
          const exists = attachments.some((a) => canonical((a as any).href || (a as any).url) === canonical(previewHref))
          if (!exists) attachments.unshift({ type: 'image', url: previewImgUrl, href: previewHref, variant: 'preview' })
        }
      } catch {}
      // Fallback: if no raster/SVG outputs were uploaded, embed the composed SVG directly if available
      if (attachments.length === 0 && json?.composition?.svg) {
        try {
          const svg = json.composition.svg as string
          const dataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
          attachments = [{ type: 'image', url: dataUrl }]
        } catch {}
      }

      // Ensure the original uploaded image is included as an attachment if missing
      try {
        const usedAnalysis = (json && json.analysis) ? json.analysis : (analysisForSend || activeConv.analysis)
        const originalRaw = usedAnalysis?.original_url_internal || usedAnalysis?.original_url || usedAnalysis?.original_data_url
        if (originalRaw) {
          const origHref = finalizeUrl(normalizeUrl(originalRaw))
          const hasOrig = attachments.some((a) => canonical((a as any).url) === canonical(origHref))
          if (!hasOrig) {
            // Prefer to place the original image right after the preview (if present)
            const previewIndex = attachments.findIndex((a) => (a as any).variant === 'preview')
            if (previewIndex >= 0) {
              attachments.splice(previewIndex + 1, 0, { type: 'image', url: finalizeDirect(origHref), href: origHref, variant: 'original' })
            } else {
              // If no preview exists, place original at the front
              attachments.unshift({ type: 'image', url: finalizeDirect(origHref), href: origHref, variant: 'original' })
            }
          }
        }
      } catch {}

      const assistantMsg: Message = {
        id: uid(),
        role: "assistant",
        content: textParts.join("\n\n"),
        timestamp: new Date().toISOString(),
        attachments,
      }

      updateActive((c) => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        analysis: json.analysis ? json.analysis : (c.analysis || analysisForSend || null),
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
          { id: uid(), role: "assistant", content: `Error: ${err?.message || String(err)}`, timestamp: new Date().toISOString() },
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

  // Persist active conversation when it changes (requires Google session)
  useEffect(() => {
    const persist = async () => {
      try {
        if (!session) return
        const current = conversations.find((c) => c.id === activeId)
        if (!current) return
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';
        await fetch(`${apiUrl}/api/conversations/${current.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(current),
        })
      } catch {}
    }
    persist()
  }, [conversations, activeId, session])

  // Backup all conversations to localStorage so refreshes survive dev server resets
  useEffect(() => {
    try {
      const key = `mw:conversations:${session?.user?.email || 'anon'}`
      localStorage.setItem(key, JSON.stringify(conversations))
    } catch {}
  }, [conversations, session])

  // Load conversations for this user on mount/login
  useEffect(() => {
    const load = async () => {
      try {
        let loaded = false
        if (session) {
          try {
            const res = await fetch('/api/conversations', { cache: 'no-store' })
            if (res.ok) {
              const data = await res.json()
              const list: Conversation[] = data?.conversations || []
              if (list.length > 0) {
                setConversations(list)
                setActiveId(list[0].id)
                loaded = true
              }
            }
          } catch {}
        }

        if (!loaded) {
          // Fallback: load from localStorage (also supports non-auth users)
          try {
            const key = `mw:conversations:${session?.user?.email || 'anon'}`
            const raw = typeof window !== 'undefined' ? localStorage.getItem(key) : null
            if (raw) {
              const cached = JSON.parse(raw)
              if (Array.isArray(cached) && cached.length > 0) {
                setConversations(cached)
                setActiveId(cached[0].id)
                // If logged in, sync back to server in background
                if (session) {
                  ;(async () => {
                    try {
                      await Promise.allSettled(
                        cached.slice(0, 10).map((c: Conversation) =>
                          fetch(`/api/conversations/${c.id}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(c),
                          })
                        )
                      )
                    } catch {}
                  })()
                }
              }
            }
          } catch {}
        }
      } catch {}
    }
    load()
  }, [session])

  // Auto-ingest pending data handed off from Home and send once
  const handleSendWithPending = useCallback(async () => {
    if (autoRanRef.current) return
    const hasPending = !!pendingPrompt || (pendingFiles && pendingFiles.length > 0)
    if (!hasPending) return

    autoRanRef.current = true
    const firstFile = pendingFiles?.[0] || null
    setImageFile(firstFile)
    
    // Set the input to the pending prompt
    if (pendingPrompt) {
      setInput(pendingPrompt)
    }
    
    // Allow state to update before sending
    await new Promise((r) => setTimeout(r, 100))
    
    // Only send if we have both an image and a prompt
    if (firstFile && pendingPrompt) {
      await handleSendMessage(new Event('submit') as any)
    }
    
    // Clear pending state
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

  // Measure composer height so we can pad the scroll area accordingly
  const composerRef = useRef<HTMLDivElement>(null)
  const [composerH, setComposerH] = useState<number>(0)
  useEffect(() => {
    const update = () => {
      try { setComposerH(composerRef.current?.offsetHeight || 0) } catch {}
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  // If any assistant message contains a 'preview' attachment, switch to wide layout (hide sidebar)
  const hasPreview = !!activeConv.messages.find((m) => m.role === 'assistant' && (m.attachments || []).some((a: any) => (a?.variant === 'preview')))

  return (
    <div className={`h-full min-h-0 w-full bg-transparent overflow-visible flex ${hasPreview ? 'pt-0' : 'pt-4 md:pt-6'}`}>
      {/* Left: Conversations (hidden when a preview is present for a full-width ad view) */}
      {!hasPreview && (
      <aside className="hidden md:flex w-[220px] lg:w-[260px] flex-col">
        <div className="p-3">
          <div className="relative group isolate">
            <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
            <Button onClick={startNewChat} className="relative z-10 w-full h-16 rounded-2xl border border-zinc-800 bg-black/60 hover:bg-black/70 text-lg transition-all shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500/40">
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
                  className={`relative z-10 w-full text-left px-4 py-2.5 text-sm rounded-xl border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500/30 ${
                    activeId === c.id 
                      ? 'border-zinc-700/60 bg-zinc-900 hover:bg-zinc-900' 
                      : 'border-zinc-800 bg-black/40 hover:bg-black/50'
                  }`}
                  aria-current={activeId === c.id ? 'true' : undefined}
                >
                  {activeId === c.id && (
                    <span className="pointer-events-none absolute left-2 top-2 bottom-2 w-1 rounded-full bg-gradient-to-b from-zinc-400 via-zinc-500 to-zinc-600 opacity-70" />
                  )}
                  <div className="truncate text-sm font-medium text-zinc-200 pl-1">{c.title || "Untitled"}</div>
                  <div className="text-xs text-zinc-500 pl-1">{c.messages.length} messages</div>
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>
      )}

      {/* Vertical separator */}
      <div aria-hidden="true" className="hidden" />

      {/* Right: Chat */}
      <section className="flex-1 min-h-0 flex flex-col">
        {/* Messages & Collage */}
        <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain chat-scrollbar no-anchor w-full">
          <div className="mx-auto w-full max-w-3xl px-4 md:px-6 py-6 space-y-8" style={{ paddingBottom: Math.max(64, composerH + 24) }}>
            {activeConv.messages.length === 0 && (
              <div className="w-full">
                <div className="mt-2 grid grid-cols-2 gap-3">
                  {suggestions4.map((s, i) => (
                    <div key={i} className="relative group isolate">
                      <div className="pointer-events-none absolute -inset-[2px] rounded-xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
                      <button
                        onClick={() => {
                          if (s.preset) {
                            if (s.preset.platform) setPlatform(s.preset.platform as any)
                            if (s.preset.tone) setTone(s.preset.tone as any)
                            if (s.preset.aspect) setSelectedAspectRatio(s.preset.aspect)
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
                <div className="relative group isolate w-full max-w-3xl">
                  <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0"></div>
                  <div className="relative z-10 rounded-2xl px-5 py-3.5 border bg-zinc-900/50 border-zinc-900/60">
                    <div className="whitespace-pre-wrap text-zinc-200 text-[15px]">{m.content}</div>
                    {m.attachments && m.attachments.length > 0 && (
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        {m.attachments.map((a, i) => (
                          <a key={i} href={deproxyUrl((a as any).href || a.url)} target="_blank" rel="noreferrer" className="block">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={a.url}
                              data-href={deproxyUrl((a as any).href || a.url)}
                              alt="attachment"
                              className="rounded-md border border-zinc-900/60 max-h-48 object-contain"
                              onLoad={(e) => {
                                const img = e.currentTarget
                                console.info("img loaded (user)", { src: img.currentSrc || img.src, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight })
                              }}
                              onError={async (e) => {
                                const img = e.currentTarget as HTMLImageElement
                                try {
                                  const tried = img.getAttribute('data-fallback') || '0'
                                  const href = img.dataset.href || ''
                                  if (tried === '0') {
                                    img.setAttribute('data-fallback', '1')
                                    img.src = deproxyUrl(href)
                                    return
                                  }
                                  if (tried === '1') {
                                    img.setAttribute('data-fallback', '2')
                                    const res = await fetch(deproxyUrl(href), { cache: 'no-store' })
                                    const blob = await res.blob()
                                    const obj = URL.createObjectURL(blob)
                                    img.src = obj
                                    return
                                  }
                                } catch {}
                                console.error('img failed (user) - giving up', { src: img.src })
                              }}
                            />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start">
                <div className="relative group isolate w-full">
                  {!m.attachments?.some((a:any)=>a?.variant==='preview') && (
                    <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-5 group-hover:opacity-10 blur-sm transition-opacity z-0"></div>
                  )}
                  <div className={`relative z-10 rounded-2xl ${m.attachments?.some((a:any)=>a?.variant==='preview') ? 'px-0 py-0 border-none bg-transparent max-w-3xl mx-auto' : 'px-5 py-3.5 border bg-zinc-900/40 border-zinc-900/60 max-w-3xl mx-auto'}`}>
                    {m.content && !m.attachments?.some((a:any)=>a?.variant==='preview') && (
                      <div className="whitespace-pre-wrap text-zinc-200 text-[15px] mb-2 max-w-screen-lg">{m.content}</div>
                    )}
                    {/* When a preview exists, hide the assistant text block to avoid misalignment */}
                    {m.attachments && m.attachments.length > 0 && (
                      <div className="space-y-4">
                        {/* Preview image - show first and larger */}
                        {m.attachments.find(a => (a as any).variant === 'preview') && (
                          <div className="w-full">
                            <a 
                              href={deproxyUrl(((m.attachments?.find(a => (a as any).variant === 'preview') as any)?.href || m.attachments?.find(a => (a as any).variant === 'preview')?.url) || '')} 
                              target="_blank" 
                              rel="noreferrer" 
                              className="relative block w-full max-w-3xl mx-auto"
                            >
                              {/* Use processed attachment.url so SVGs are already inlined as data URLs */}
                              {
                                (() => {
                                  const prev = m.attachments?.find(a => (a as any).variant === 'preview') as any
                                  const src = (prev?.url || '') as string
                                  return (
                                    // eslint-disable-next-line @next/next/no-img-element
                                    <img
                                      src={src}
                                      data-href={deproxyUrl((prev?.href || prev?.url || '') as string)}
                                      alt="Preview"
                                      className="relative z-10 block rounded-lg w-full h-auto max-h-[58vh] object-contain bg-transparent"
                                      onLoad={() => {
                                        try { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) } catch {}
                                      }}
                                      onError={async (e) => {
                                        const img = e.currentTarget as HTMLImageElement
                                        try {
                                          const tried = img.getAttribute('data-fallback') || '0'
                                          const href = img.dataset.href || ''
                                          if (tried === '0') { img.setAttribute('data-fallback', '1'); img.src = deproxyUrl(href); return }
                                          if (tried === '1') {
                                            img.setAttribute('data-fallback', '2')
                                            const res = await fetch(deproxyUrl(href), { cache: 'no-store' })
                                            const blob = await res.blob()
                                            img.src = URL.createObjectURL(blob)
                                            return
                                          }
                                        } catch {}
                                      }}
                                    />
                                  )
                                })()
                              }
                            </a>
                          </div>
                        )}
                        
                        {/* Other attachments displayed directly in a clean responsive grid */}
                        {m.attachments.some(a => (a as any).variant !== 'preview') && (
                          <div className="max-w-3xl mx-auto">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                              {m.attachments
                                .filter(a => (a as any).variant !== 'preview')
                                .map((a, i) => (
                                  <a 
                                    key={i} 
                                    href={deproxyUrl((a as any).href || (a as any).url)} 
                                    target="_blank" 
                                    rel="noreferrer" 
                                    className="block rounded-md overflow-hidden transition"
                                  >
                                    <div className="relative w-full aspect-[4/3] bg-transparent">
                                      {/* eslint-disable-next-line @next/next/no-img-element */}
                                      <img
                                        src={(a as any).url}
                                        data-href={deproxyUrl((a as any).href || (a as any).url)}
                                        alt={`${(a as any).variant || 'output'} ${i+1}`}
                                        className="absolute inset-0 w-full h-full object-cover"
                                        onLoad={(e) => {
                                          const img = e.currentTarget
                                          console.info("img loaded (assistant)", { src: img.currentSrc || img.src })
                                        }}
                                        onError={async (e) => {
                                          const img = e.currentTarget as HTMLImageElement
                                          try {
                                            const tried = img.getAttribute('data-fallback') || '0'
                                            const href = img.dataset.href || ''
                                            if (tried === '0') { img.setAttribute('data-fallback', '1'); img.src = deproxyUrl(href); return }
                                            if (tried === '1') {
                                              img.setAttribute('data-fallback', '2')
                                              const res = await fetch(deproxyUrl(href), { cache: 'no-store' })
                                              const blob = await res.blob()
                                              img.src = URL.createObjectURL(blob)
                                              return
                                            }
                                          } catch {}
                                          console.error('img failed (assistant) - giving up', { src: img.src })
                                        }}
                                      />
                                    </div>
                                  </a>
                              ))}
                            </div>
                          </div>
                        )}
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
                <div className="relative group isolate max-w-full">
                  <div className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#6366f1_0deg,#ec4899_120deg,#f59e0b_240deg,#6366f1_360deg)] opacity-5 blur-sm transition-opacity z-0"></div>
                  <div className="relative z-10 rounded-2xl px-5 py-4 border bg-zinc-900/40 border-zinc-900/60">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Anchor for auto-scroll at the end of messages */}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Composer (raised, levitating with glow) */}
        <div ref={composerRef} className={`fixed bottom-8 right-0 z-40 p-4 pb-[env(safe-area-inset-bottom)] bg-transparent ${!hasPreview ? 'left-0 md:left-[220px] lg:left-[260px]' : 'left-0'}`} onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
          <div className="mx-auto w-full max-w-3xl px-4 md:px-6">
            <div className="relative w-full max-w-3xl mx-auto">
            {/* Ambient levitation glow */}
              <div className="pointer-events-none absolute -inset-8 rounded-3xl bg-[radial-gradient(60%_60%_at_50%_50%,rgba(99,102,241,0.28),rgba(236,72,153,0.24)_50%,rgba(245,158,11,0.2)_80%,transparent_100%)] blur-3xl"></div>
              <div className="mb-2 flex justify-end">
                <CreditsPill variant="inline" />
              </div>
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
                          className="relative h-12 rounded-full bg-zinc-900/70 hover:bg-zinc-900 text-zinc-200 border border-zinc-700/60 flex items-center justify-center px-4 gap-2 shadow-[0_0_22px_rgba(236,72,153,0.25)]"
                          title="Options"
                        >
                          <SlidersHorizontal className="w-4 h-4" />
                          <span className="text-sm font-medium">Options</span>
                        </button>
                      </PopoverTrigger>
                      <PopoverContent align="end" className="w-80 bg-zinc-950 border-zinc-900 text-zinc-200">
                        <div className="space-y-3">
                          <div>
                            <div className="text-[11px] uppercase tracking-wide text-zinc-400 mb-1">Platform</div>
                            <ToggleGroup type="single" value={platform} onValueChange={(v) => setPlatform((v as any) || "") } className="bg-zinc-900/40 border border-zinc-900 rounded-md">
                              <ToggleGroupItem value="instagram" aria-label="Instagram" className="px-2 text-xs">IG</ToggleGroupItem>
                              <ToggleGroupItem value="facebook" aria-label="Facebook" className="px-2 text-xs">FB</ToggleGroupItem>
                              <ToggleGroupItem value="linkedin" aria-label="LinkedIn" className="px-2 text-xs">LI</ToggleGroupItem>
                              <ToggleGroupItem value="x" aria-label="X" className="px-2 text-xs">X</ToggleGroupItem>
                            </ToggleGroup>
                          </div>
                          <div>
                            <div className="text-[11px] uppercase tracking-wide text-zinc-400 mb-1">Tone</div>
                            <ToggleGroup type="single" value={tone} onValueChange={(v) => setTone((v as any) || "") } className="bg-zinc-900/40 border border-zinc-900 rounded-md">
                              <ToggleGroupItem value="bold" aria-label="Bold" className="px-2 text-xs">B</ToggleGroupItem>
                              <ToggleGroupItem value="friendly" aria-label="Friendly" className="px-2 text-xs">F</ToggleGroupItem>
                              <ToggleGroupItem value="professional" aria-label="Professional" className="px-2 text-xs">P</ToggleGroupItem>
                              <ToggleGroupItem value="minimal" aria-label="Minimal" className="px-2 text-xs">M</ToggleGroupItem>
                            </ToggleGroup>
                          </div>
                          <div>
                            <div className="text-[11px] uppercase tracking-wide text-zinc-400 mb-1">Dimensions</div>
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
                          <div>
                            <div className="text-[11px] uppercase tracking-wide text-zinc-400 mb-1">Text color override</div>
                            <div className="flex items-center gap-2">
                              <Input
                                type="text"
                                inputMode="text"
                                placeholder="#ffffff"
                                value={textColorOverride}
                                onChange={(e) => setTextColorOverride(e.target.value)}
                                className="h-8"
                              />
                              <input
                                type="color"
                                value={/^#([0-9a-fA-F]{6})$/.test(textColorOverride || "") ? textColorOverride : "#ffffff"}
                                onChange={(e) => setTextColorOverride(e.target.value)}
                                aria-label="Pick color"
                                className="h-8 w-8 rounded border border-zinc-800 bg-transparent p-0"
                                style={{ padding: 0 }}
                              />
                            </div>
                            <div className="text-[10px] text-zinc-500 mt-1">Optional. Hex like #ffffff or #fff</div>
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
                        <div className="relative z-10 w-12 h-12 rounded-full bg-black/70 hover:bg-black border border-zinc-700 flex items-center justify-center">
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
        </div>
      </section>
    </div>
  )
}
