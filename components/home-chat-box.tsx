"use client"

import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { useApp } from "./app-context"
import { PlusCircle, Send, ImageIcon } from "lucide-react"

export function HomeChatBox() {
  const { setPendingPrompt, setPendingFiles } = useApp()
  const [prompt, setPrompt] = useState("")
  const [files, setFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const onFiles = (incoming: FileList | File[]) => {
    const imgs = Array.from(incoming).filter((f) => f.type.startsWith("image/"))
    setFiles(imgs)
  }

  const handleSubmit = () => {
    if (files.length === 0) {
      fileInputRef.current?.click()
      return
    }
    setPendingFiles(files)
    setPendingPrompt(prompt)
    // switch to upload view by navigating to hash that app/page uses
    window.scrollTo({ top: 0, behavior: "smooth" })
    const uploadTab = document.querySelector('[data-nav="upload"]') as HTMLElement | null
    uploadTab?.click()
  }

  return (
    <div className="rounded-3xl bg-zinc-950/70 border border-zinc-900 backdrop-blur p-4">
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Button onClick={() => fileInputRef.current?.click()} className="rounded-full">
              <PlusCircle className="w-4 h-4 mr-2" /> Add image
            </Button>
            <div className="text-xs text-zinc-400">JPG, PNG, WebP • Max 10MB</div>
          </div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the ad you want (tone, facts, CTA). We never invent facts."
            className="w-full p-4 min-h-20 rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-200 placeholder-zinc-500"
          />
        </div>
        <Button onClick={handleSubmit} disabled={files.length === 0} className="rounded-full px-6 py-3">
          <Send className="w-4 h-4 mr-2" /> Generate
        </Button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="image/*"
        multiple
        onChange={(e) => e.target.files && onFiles(e.target.files)}
      />
      {files.length > 0 && (
        <div className="mt-3 flex gap-3 overflow-x-auto">
          {files.map((f, i) => (
            <div key={i} className="w-20 h-20 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-500 text-xs">
              <ImageIcon className="w-4 h-4 mr-1" /> {f.name.slice(0, 8)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


