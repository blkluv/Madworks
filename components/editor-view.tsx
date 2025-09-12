"use client"

import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useSession } from "next-auth/react"
import { cn } from "@/lib/utils"
import { Image as ImageIcon, Type, Trash2, Download, Palette, Plus, AlignLeft, AlignCenter, AlignRight, Layers } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Basic design element model
interface BaseEl {
  id: string
  x: number
  y: number
  width: number
  height: number
  rotation?: number
}

// Built-in icon set (SVG strings)
const ICONS: { name: string; svg: string }[] = [
  { name: 'Star', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M11.48 3.5c.2-.62 1.04-.62 1.24 0l1.7 5.26h5.53c.65 0 .92.83.4 1.21l-4.47 3.25 1.7 5.26c.2.62-.51 1.14-1.04.76L12.1 15.5l-4.44 3.25c-.53.38-1.24-.14-1.04-.76l1.7-5.26L3.86 9.97c-.52-.38-.25-1.21.4-1.21h5.53l1.7-5.26Z"/></svg>' },
  { name: 'Heart', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M11.99 20.3 4.7 13.02a5.25 5.25 0 1 1 7.42-7.42l.87.88.87-.88a5.25 5.25 0 1 1 7.42 7.42L11.99 20.3Z"/></svg>' },
  { name: 'Check', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>' },
  { name: 'Arrow Right', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>' },
  { name: 'Sparkles', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M8 2.5c.2-.5.8-.5 1 0L10.7 6 14 7.3c.5.2.5.8 0 1L10.7 9.6 9 13c-.2.5-.8.5-1 0L6.3 9.6 3 8.3c-.5-.2-.5-.8 0-1L6.3 6 8 2.5ZM18 10.5c.2-.5.8-.5 1 0l.8 1.9 1.9.8c.5.2.5.8 0 1l-1.9.8L19 17c-.2.5-.8.5-1 0l-.8-1.9-1.9-.8c-.5-.2-.5-.8 0-1l1.9-.8.8-1.9Z"/></svg>' },
  { name: 'Circle', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="8"/></svg>' },
  { name: 'Square', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>' },
  { name: 'Triangle', svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 5 20 19H4L12 5Z"/></svg>' },
]

function iconToDataUrl(svg: string, color = '#ffffff') {
  // Replace currentColor with requested color
  const colored = svg.replace(/currentColor/g, color)
  return 'data:image/svg+xml;utf8,' + encodeURIComponent(colored)
}

interface TextEl extends BaseEl {
  kind: "text"
  text: string
  fontSize: number
  fontFamily: string
  fontWeight: 300 | 400 | 500 | 600 | 700 | 800 | 900
  color: string
  align: "left" | "center" | "right"
  background?: string
}

interface ImageEl extends BaseEl {
  kind: "image"
  url: string
}

type El = TextEl | ImageEl

function uid(prefix = "el"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`
}

function clamp(n: number, min: number, max: number) { return Math.max(min, Math.min(max, n)) }

function usePointerDrag(onMove: (dx: number, dy: number, e: PointerEvent) => void, onEnd?: () => void) {
  const moveRef = useRef(onMove)
  const endRef = useRef(onEnd)
  moveRef.current = onMove
  endRef.current = onEnd

  return useCallback((e: React.PointerEvent) => {
    const startX = e.clientX
    const startY = e.clientY
    const target = e.currentTarget as HTMLElement
    target.setPointerCapture(e.pointerId)

    function onPointerMove(ev: PointerEvent) {
      moveRef.current?.(ev.clientX - startX, ev.clientY - startY, ev)
    }
    function onPointerUp() {
      try { target.releasePointerCapture(e.pointerId) } catch {}
      window.removeEventListener("pointermove", onPointerMove)
      window.removeEventListener("pointerup", onPointerUp)
      endRef.current?.()
    }

    window.addEventListener("pointermove", onPointerMove)
    window.addEventListener("pointerup", onPointerUp)
  }, [])
}

function useConversationsImages() {
  const { data: session } = useSession()
  const [images, setImages] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const found: string[] = []
      try {
        const res = await fetch('/api/conversations', { cache: 'no-store' })
        if (res.ok) {
          const data = await res.json()
          const list = (data?.conversations || []) as Array<any>
          for (const c of list) {
            for (const m of (c?.messages || [])) {
              for (const a of (m?.attachments || [])) {
                if (a?.type === 'image' && a?.url) found.push(a.url)
              }
            }
          }
        }
      } catch {}

      if (found.length === 0) {
        try {
          const key = `mw:conversations:${session?.user?.email || 'anon'}`
          const raw = typeof window !== 'undefined' ? localStorage.getItem(key) : null
          if (raw) {
            const cached = JSON.parse(raw)
            for (const c of (Array.isArray(cached) ? cached : [])) {
              for (const m of (c?.messages || [])) {
                for (const a of (m?.attachments || [])) {
                  if (a?.type === 'image' && a?.url) found.push(a.url)
                }
              }
            }
          }
        } catch {}
      }

      const clean = Array.from(new Set(found.filter(Boolean))).filter((u) => typeof u === 'string')
      if (!cancelled) setImages(clean)
    })()
    return () => { cancelled = true }
  }, [session])

  return images
}

async function getImageNaturalSize(url: string): Promise<{ w: number; h: number } | null> {
  try {
    await new Promise<void>((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = "anonymous"
      img.onload = () => resolve()
      img.onerror = () => resolve() // resolve even on error
      img.src = url
    })
    return new Promise((resolve) => {
      const i = new Image()
      i.onload = () => resolve({ w: i.naturalWidth || 800, h: i.naturalHeight || 800 })
      i.onerror = () => resolve({ w: 800, h: 800 })
      i.src = url
    })
  } catch {
    return null
  }
}

export function EditorView() {
  const imagesFromChats = useConversationsImages()
  const [uploads, setUploads] = useState<string[]>([])
  const [elements, setElements] = useState<El[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [bg, setBg] = useState<string>("#0a0a0a")
  const [aspect, setAspect] = useState("1:1")
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [scale, setScale] = useState<number>(1)
  const [assetTab, setAssetTab] = useState<'images'|'icons'|'uploads'>('images')
  const [assetQuery, setAssetQuery] = useState('')

  const canvasPx = useMemo(() => {
    switch (aspect) {
      case "4:5": return { w: 864, h: 1080 }
      case "9:16": return { w: 675, h: 1200 }
      case "16:9": return { w: 1280, h: 720 }
      case "1:1":
      default: return { w: 1000, h: 1000 }
    }
  }, [aspect])

  // Load previous design
  useEffect(() => {
    try {
      const raw = localStorage.getItem('mw:editor:current')
      if (raw) {
        const parsed = JSON.parse(raw)
        setElements(parsed?.elements || [])
        setBg(parsed?.bg || bg)
        setAspect(parsed?.aspect || aspect)
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist design
  useEffect(() => {
    try {
      localStorage.setItem('mw:editor:current', JSON.stringify({ elements, bg, aspect }))
    } catch {}
  }, [elements, bg, aspect])

  // Fit canvas to available width on small screens by scaling down
  useEffect(() => {
    const updateScale = () => {
      try {
        const maxW = containerRef.current?.clientWidth || canvasPx.w
        const next = Math.min(1, Math.max(0.2, (maxW - 2) / canvasPx.w))
        setScale(next)
      } catch {}
    }
    updateScale()
    window.addEventListener('resize', updateScale)
    return () => window.removeEventListener('resize', updateScale)
  }, [canvasPx.w])

  const selected = elements.find((e) => e.id === selectedId) || null

  // Add text element
  const addText = () => {
    const id = uid('text')
    const el: TextEl = {
      id,
      kind: 'text',
      x: 80,
      y: 80,
      width: 360,
      height: 100,
      text: 'Add your headline...'
        , fontSize: 36,
      fontFamily: 'Inter, ui-sans-serif, system-ui',
      fontWeight: 700,
      color: '#ffffff',
      align: 'left',
    }
    setElements((prev) => [...prev, el])
    setSelectedId(id)
  }

  // Add image element
  const addImage = async (url: string) => {
    const id = uid('img')
    let w = 400, h = 300
    const sz = await getImageNaturalSize(url)
    if (sz) {
      const maxW = Math.min(canvasPx.w * 0.6, 700)
      const scale = Math.min(1, maxW / sz.w)
      w = Math.max(120, Math.round(sz.w * scale))
      h = Math.max(120, Math.round(sz.h * scale))
    }
    const el: ImageEl = { id, kind: 'image', url, x: 120, y: 120, width: w, height: h }
    setElements((prev) => [...prev, el])
    setSelectedId(id)
  }

  // Add icon element (as image from SVG data URL)
  const addIcon = async (svg: string) => {
    const url = iconToDataUrl(svg, '#ffffff')
    const id = uid('icon')
    const el: ImageEl = { id, kind: 'image', url, x: 140, y: 140, width: 120, height: 120 }
    setElements((prev) => [...prev, el])
    setSelectedId(id)
  }

  const onUploadFile = (file: File) => {
    const url = URL.createObjectURL(file)
    setUploads((u) => [url, ...u])
    addImage(url)
  }

  // Bring forward/back
  const bringToFront = () => {
    if (!selected) return
    setElements((prev) => {
      const idx = prev.findIndex((e) => e.id === selected.id)
      if (idx < 0) return prev
      const copy = [...prev]
      const [el] = copy.splice(idx, 1)
      copy.push(el)
      return copy
    })
  }
  const sendToBack = () => {
    if (!selected) return
    setElements((prev) => {
      const idx = prev.findIndex((e) => e.id === selected.id)
      if (idx < 0) return prev
      const copy = [...prev]
      const [el] = copy.splice(idx, 1)
      copy.unshift(el)
      return copy
    })
  }

  // Reorder specific layer by id
  const bringForwardById = (id: string) => {
    setElements((prev) => {
      const idx = prev.findIndex((e) => e.id === id)
      if (idx < 0 || idx === prev.length - 1) return prev
      const copy = [...prev]
      const [el] = copy.splice(idx, 1)
      copy.splice(idx + 1, 0, el)
      return copy
    })
  }

  const sendBackwardById = (id: string) => {
    setElements((prev) => {
      const idx = prev.findIndex((e) => e.id === id)
      if (idx <= 0) return prev
      const copy = [...prev]
      const [el] = copy.splice(idx, 1)
      copy.splice(idx - 1, 0, el)
      return copy
    })
  }

  // Delete element
  const deleteSelected = () => {
    if (!selected) return
    setElements((prev) => prev.filter((e) => e.id !== selected.id))
    setSelectedId((id) => (id === selected?.id ? null : id))
  }

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selected) {
          e.preventDefault()
          deleteSelected()
        }
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') { // duplicate
        const el = selected
        if (el) {
          e.preventDefault()
          const copy = { ...el, id: uid(el.kind === 'text' ? 'text' : 'img'), x: el.x + 20, y: el.y + 20 } as El
          setElements((prev) => [...prev, copy])
          setSelectedId(copy.id)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected])

  // Drag logic
  const startDrag = (id: string) => usePointerDrag((dx, dy) => {
    const s = scale || 1
    const ddx = dx / s
    const ddy = dy / s
    setElements((prev) => prev.map((e) => e.id === id ? { ...e, x: clamp(e.x + ddx, 0, canvasPx.w - e.width), y: clamp(e.y + ddy, 0, canvasPx.h - e.height) } : e))
  })

  type Corner = 'nw' | 'ne' | 'sw' | 'se'
  const startResize = (id: string, corner: Corner) => usePointerDrag((dx, dy) => {
    const s = scale || 1
    const ddx = dx / s
    const ddy = dy / s
    setElements((prev) => prev.map((e) => {
      if (e.id !== id) return e
      let { x, y, width, height } = e
      const min = 40
      if (corner === 'se') { width = clamp(width + ddx, min, canvasPx.w - x); height = clamp(height + ddy, min, canvasPx.h - y) }
      if (corner === 'sw') { width = clamp(width - ddx, min, x + width); height = clamp(height + ddy, min, canvasPx.h - y); x = clamp(x + ddx, 0, x + width) }
      if (corner === 'ne') { width = clamp(width + ddx, min, canvasPx.w - x); height = clamp(height - ddy, min, y + height); y = clamp(y + ddy, 0, y + height) }
      if (corner === 'nw') { width = clamp(width - ddx, min, x + width); height = clamp(height - ddy, min, y + height); x = clamp(x + ddx, 0, x + width); y = clamp(y + ddy, 0, y + height) }
      return { ...e, x, y, width, height }
    }))
  })

  // Inspector updates
  const updateText = (patch: Partial<TextEl>) => {
    if (!selected || (selected as El).kind !== 'text') return
    setElements((prev) => prev.map((e) => e.id === selected.id ? { ...(e as TextEl), ...(patch as any) } : e))
  }

  const onCanvasClick = (e: React.MouseEvent) => {
    if (e.target === wrapRef.current) setSelectedId(null)
  }

  // Export current design as PNG by re-drawing into an offscreen canvas
  const exportPNG = async () => {
    const c = document.createElement('canvas')
    c.width = canvasPx.w
    c.height = canvasPx.h
    const ctx = c.getContext('2d')
    if (!ctx) return
    // background
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, c.width, c.height)
    // draw layers in order
    for (const el of elements) {
      if ((el as any).kind === 'image') {
        const url = (el as ImageEl).url
        await new Promise<void>((resolve) => {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.onload = () => { ctx.drawImage(img, el.x, el.y, el.width, el.height); resolve() }
          img.onerror = () => resolve()
          img.src = url
        })
      } else {
        const t = el as TextEl
        ctx.fillStyle = t.color
        ctx.textAlign = t.align
        ctx.textBaseline = 'top'
        ctx.font = `${t.fontWeight} ${t.fontSize}px ${t.fontFamily}`
        // simple line wrapping within element width
        const words = (t.text || '').split(/\s+/)
        let line = ''
        let y = el.y
        const maxW = el.width
        const originX = t.align === 'left' ? el.x : t.align === 'center' ? el.x + maxW / 2 : el.x + maxW
        for (const w of words) {
          const test = line ? line + ' ' + w : w
          const m = ctx.measureText(test)
          if (m.width > maxW && line) {
            ctx.fillText(line, originX, y, maxW)
            y += t.fontSize * 1.3
            line = w
          } else {
            line = test
          }
        }
        if (line) ctx.fillText(line, originX, y, maxW)
      }
    }
    const url = c.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url
    a.download = `madworks-design-${Date.now()}.png`
    a.click()
  }

  // Export as JSON (design data)
  const exportJSON = () => {
    const data = { bg, aspect, elements }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `madworks-design-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-0">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500">
            <ImageIcon className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-2xl font-semibold">Studio</h1>
        </div>
        <div className="flex items-center gap-2">
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="rounded-xl border-zinc-800">
                <Palette className="w-4 h-4 mr-2" /> Background
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-56 bg-zinc-950 border-zinc-900">
              <div className="space-y-2">
                <Input type="color" value={bg} onChange={(e) => setBg(e.target.value)} className="h-10 p-1" />
                <Select value={aspect} onValueChange={setAspect}>
                  <SelectTrigger className="bg-zinc-900 border-zinc-800 rounded-xl">
                    <SelectValue placeholder="Aspect" />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-950 border-zinc-900">
                    <SelectItem value="1:1">Square (1:1)</SelectItem>
                    <SelectItem value="4:5">Portrait (4:5)</SelectItem>
                    <SelectItem value="9:16">Story (9:16)</SelectItem>
                    <SelectItem value="16:9">Landscape (16:9)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </PopoverContent>
          </Popover>
          <Button onClick={addText} className="rounded-xl">
            <Type className="w-4 h-4 mr-2" /> Add Text
          </Button>
          <label className="inline-flex items-center">
            <input type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onUploadFile(f) }} />
            <Button className="rounded-xl" asChild>
              <span><Plus className="w-4 h-4 mr-2" /> Add Image</span>
            </Button>
          </label>
          <Button variant="outline" onClick={exportPNG} className="rounded-xl border-zinc-800">
            <Download className="w-4 h-4 mr-2" /> Export PNG
          </Button>
          <Button variant="outline" onClick={exportJSON} className="rounded-xl border-zinc-800">
            <Download className="w-4 h-4 mr-2" /> Export JSON
          </Button>
          {selected && (
            <>
              <Button variant="outline" onClick={bringToFront} className="rounded-xl border-zinc-800"><Layers className="w-4 h-4 mr-2" />Front</Button>
              <Button variant="outline" onClick={sendToBack} className="rounded-xl border-zinc-800">Back</Button>
              <Button variant="destructive" onClick={deleteSelected} className="rounded-xl"><Trash2 className="w-4 h-4 mr-2" />Delete</Button>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
        {/* Assets */}
        <aside className="md:col-span-3 order-2 md:order-1 bg-zinc-950/80 border border-zinc-900 rounded-2xl p-3 h-[72vh] overflow-y-auto">
          <Tabs value={assetTab} onValueChange={(v) => setAssetTab(v as any)} className="w-full">
            <TabsList className="grid grid-cols-3 bg-zinc-900 rounded-xl">
              <TabsTrigger value="images">Images</TabsTrigger>
              <TabsTrigger value="icons">Icons</TabsTrigger>
              <TabsTrigger value="uploads">Uploads</TabsTrigger>
            </TabsList>
            <div className="mt-3">
              <Input placeholder="Search assets" value={assetQuery} onChange={(e) => setAssetQuery(e.target.value)} className="h-8 bg-zinc-900 border-zinc-800" />
            </div>
            <TabsContent value="images" className="mt-3">
              <h3 className="text-sm font-medium text-zinc-300 mb-2">From chats</h3>
              <div className="grid grid-cols-2 gap-2">
                {imagesFromChats.filter(u => !assetQuery || u.toLowerCase().includes(assetQuery.toLowerCase())).length === 0 && (
                  <div className="col-span-2 text-xs text-zinc-500">No images found.</div>
                )}
                {imagesFromChats.filter(u => !assetQuery || u.toLowerCase().includes(assetQuery.toLowerCase())).map((u) => (
                  <button key={u} onClick={() => addImage(u)} className="group relative border border-zinc-900 rounded-lg overflow-hidden">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={u} alt="asset" className="w-full h-24 object-cover group-hover:opacity-90" />
                  </button>
                ))}
              </div>
            </TabsContent>
            <TabsContent value="icons" className="mt-3">
              <div className="grid grid-cols-3 gap-2">
                {ICONS.filter(i => !assetQuery || i.name.toLowerCase().includes(assetQuery.toLowerCase())).map((ic) => (
                  <button key={ic.name} onClick={() => addIcon(ic.svg)} className="group relative border border-zinc-900 rounded-lg p-3 bg-zinc-900/50 hover:bg-zinc-900">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img alt={ic.name} src={iconToDataUrl(ic.svg)} className="w-full h-12 object-contain" />
                    <div className="mt-2 text-[10px] text-zinc-400 text-center">{ic.name}</div>
                  </button>
                ))}
              </div>
            </TabsContent>
            <TabsContent value="uploads" className="mt-3">
              {uploads.length === 0 && (
                <div className="text-xs text-zinc-500">No uploads yet. Use "Add Image" to upload.</div>
              )}
              <div className="grid grid-cols-2 gap-2">
                {uploads.filter(u => !assetQuery || u.toLowerCase().includes(assetQuery.toLowerCase())).map((u) => (
                  <button key={u} onClick={() => addImage(u)} className="group relative border border-zinc-900 rounded-lg overflow-hidden">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={u} alt="upload" className="w-full h-24 object-cover group-hover:opacity-90" />
                  </button>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </aside>

        {/* Canvas + Inspector */}
        <main className="md:col-span-9 order-1 md:order-2 grid grid-cols-1 lg:grid-cols-9 gap-4">
          <div className="lg:col-span-6">
            <div className="rounded-2xl border border-zinc-900 bg-zinc-950/60 p-4">
              <div ref={containerRef} className="relative w-full overflow-x-auto">
                <div className="relative origin-top-left mx-auto" style={{ width: canvasPx.w, height: canvasPx.h, transform: `scale(${scale})` }}>
                  <div
                    ref={wrapRef}
                    role="region"
                    aria-label="Design canvas"
                    className="relative overflow-hidden rounded-xl"
                    style={{ width: canvasPx.w, height: canvasPx.h, background: bg }}
                    onClick={onCanvasClick}
                  >
                  {/* guidelines */}
                  <div className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2 w-px h-full bg-white/5" />
                  <div className="pointer-events-none absolute top-1/2 left-0 -translate-y-1/2 h-px w-full bg-white/5" />

                  {elements.map((el) => (
                    <ElementView
                      key={el.id}
                      el={el}
                      selected={selectedId === el.id}
                      onSelect={() => setSelectedId(el.id)}
                      onStartDrag={startDrag(el.id)}
                      onStartResize={startResize}
                      onChange={(patch) => setElements((prev) => prev.map((e) => e.id === el.id ? { ...(e as any), ...(patch as any) } : e))}
                    />
                  ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Inspector */}
          <div className="lg:col-span-3">
            <div className="rounded-2xl border border-zinc-900 bg-zinc-950/80 p-3 h-[72vh] overflow-y-auto">
              {/* Layers list */}
              <div className="mb-3">
                <h3 className="text-sm font-medium text-zinc-300 mb-2">Layers</h3>
                <div className="space-y-1">
                  {elements.length === 0 && (
                    <div className="text-xs text-zinc-500">No layers yet. Add text or images.</div>
                  )}
                  {elements.map((e, i) => (
                    <div key={e.id} className={cn("flex items-center justify-between px-2 py-1 rounded-md border cursor-pointer", selectedId===e.id?"bg-zinc-900 border-zinc-800":"bg-zinc-950 border-zinc-900")}
                      onClick={() => setSelectedId(e.id)}>
                      <div className="text-xs text-zinc-300 truncate">
                        {e.kind === 'text' ? `Text: ${(e as TextEl).text?.slice(0, 24) || '...'}` : `Image ${i+1}`}
                      </div>
                      <div className="flex gap-1">
                        <Button size="sm" variant="outline" className="h-6 px-2 border-zinc-800" onClick={(ev)=>{ev.stopPropagation(); sendBackwardById(e.id)}}>◀</Button>
                        <Button size="sm" variant="outline" className="h-6 px-2 border-zinc-800" onClick={(ev)=>{ev.stopPropagation(); bringForwardById(e.id)}}>▶</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Inspector */}
              {!selected && (
                <div className="text-sm text-zinc-400">Select a layer to edit its properties.</div>
              )}
              {selected && selected.kind === 'text' && (
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-zinc-400">Text</label>
                    <div
                      contentEditable
                      suppressContentEditableWarning
                      className="mt-1 rounded-md border border-zinc-800 bg-zinc-900/60 p-2 text-sm focus:outline-none"
                      onInput={(e) => updateText({ text: (e.target as HTMLElement).innerText || '' })}
                    >{(selected as TextEl).text}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-zinc-400">Font size</label>
                      <Input type="number" value={(selected as TextEl).fontSize}
                        onChange={(e) => updateText({ fontSize: clamp(parseInt(e.target.value || '0') || 12, 8, 300) })}
                        className="mt-1 h-8 bg-zinc-900 border-zinc-800" />
                    </div>
                    <div>
                      <label className="text-xs text-zinc-400">Weight</label>
                      <Select value={String((selected as TextEl).fontWeight)} onValueChange={(v) => updateText({ fontWeight: Number(v) as any })}>
                        <SelectTrigger className="mt-1 h-8 bg-zinc-900 border-zinc-800">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-950 border-zinc-900">
                          {[300,400,500,600,700,800,900].map(w => (
                            <SelectItem key={w} value={String(w)}>{w}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-zinc-400">Font family</label>
                    <Select value={(selected as TextEl).fontFamily} onValueChange={(v) => updateText({ fontFamily: v as any })}>
                      <SelectTrigger className="mt-1 h-8 bg-zinc-900 border-zinc-800">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-zinc-950 border-zinc-900">
                        {[
                          'Inter, ui-sans-serif, system-ui',
                          'Arial, Helvetica, sans-serif',
                          'Georgia, serif',
                          'Times New Roman, Times, serif',
                          'Courier New, monospace',
                          'Poppins, ui-sans-serif, system-ui',
                          'Montserrat, ui-sans-serif, system-ui',
                          'Lato, ui-sans-serif, system-ui',
                        ].map(f => (
                          <SelectItem key={f} value={f}>{f.split(',')[0]}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-zinc-400">Color</label>
                      <Input type="color" value={(selected as TextEl).color} onChange={(e) => updateText({ color: e.target.value })} className="mt-1 h-8 p-1" />
                    </div>
                    <div>
                      <label className="text-xs text-zinc-400">Align</label>
                      <div className="mt-1 flex gap-1">
                        <Button variant={(selected as TextEl).align==='left'?"default":"outline"} size="sm" className="rounded-md h-8 border-zinc-800" onClick={() => updateText({ align: 'left' })}><AlignLeft className="w-4 h-4"/></Button>
                        <Button variant={(selected as TextEl).align==='center'?"default":"outline"} size="sm" className="rounded-md h-8 border-zinc-800" onClick={() => updateText({ align: 'center' })}><AlignCenter className="w-4 h-4"/></Button>
                        <Button variant={(selected as TextEl).align==='right'?"default":"outline"} size="sm" className="rounded-md h-8 border-zinc-800" onClick={() => updateText({ align: 'right' })}><AlignRight className="w-4 h-4"/></Button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              {selected && selected.kind === 'image' && (
                <div className="text-sm text-zinc-400">Use the handles on the image to resize. Drag to reposition.</div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

function ElementView({ el, selected, onSelect, onStartDrag, onStartResize, onChange }: {
  el: El
  selected: boolean
  onSelect: () => void
  onStartDrag: (e: React.PointerEvent) => void
  onStartResize: (id: string, corner: 'nw' | 'ne' | 'sw' | 'se') => (e: React.PointerEvent) => void
  onChange: (patch: Partial<El>) => void
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      className={cn(
        "absolute group outline-none",
      )}
      style={{ left: el.x, top: el.y, width: el.width, height: el.height }}
      onPointerDown={(e) => { e.stopPropagation(); onSelect() }}
    >
      <div
        className={cn(
          "w-full h-full cursor-move",
          selected ? "ring-2 ring-purple-500/70" : "ring-1 ring-zinc-700/60"
        )}
        onPointerDown={(e) => { e.stopPropagation(); onStartDrag(e) }}
      >
        {el.kind === 'image' ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={(el as ImageEl).url} alt="element" className="w-full h-full object-contain select-none" draggable={false} />
        ) : (
          <div className="w-full h-full p-2 select-text"
            style={{
              color: (el as TextEl).color,
              fontWeight: (el as TextEl).fontWeight,
              fontSize: (el as TextEl).fontSize,
              fontFamily: (el as TextEl).fontFamily,
              textAlign: (el as TextEl).align,
            }}
            contentEditable
            suppressContentEditableWarning
            onInput={(e) => {
              onChange({ ...(el as any), text: (e.target as HTMLElement).innerText || '' })
            }}
          >{(el as TextEl).text}</div>
        )}
      </div>

      {/* Resize handles */}
      {selected && (
        <>
          {(["nw","ne","sw","se"] as const).map((c) => (
            <div
              key={c}
              onPointerDown={(e) => { e.stopPropagation(); onStartResize(el.id, c)(e) }}
              className={cn(
                "absolute w-3 h-3 bg-white rounded-sm border border-zinc-800", 
                c === 'nw' && "-left-1.5 -top-1.5 cursor-nwse-resize",
                c === 'ne' && "-right-1.5 -top-1.5 cursor-nesw-resize",
                c === 'sw' && "-left-1.5 -bottom-1.5 cursor-nesw-resize",
                c === 'se' && "-right-1.5 -bottom-1.5 cursor-nwse-resize",
              )}
            />
          ))}
        </>
      )}
    </div>
  )
}
