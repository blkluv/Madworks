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

  const handleSubmit = async () => {
    if (files.length === 0) {
      fileInputRef.current?.click()
      return
    }
    
    // Set pending data first
    setPendingFiles(files)
    setPendingPrompt(prompt)
    
    // Seamlessly navigate to Chat view
    window.scrollTo({ top: 0, behavior: "smooth" })
    
    // Find and click the chat tab
    const chatTab = document.querySelector('[data-nav="chat"]') as HTMLElement | null
    if (chatTab) {
      // Trigger click to switch to chat view
      chatTab.click()
    }
  }

  return (
    <div className="rounded-3xl bg-zinc-950/70 border border-zinc-900 backdrop-blur p-4">
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <Button onClick={() => fileInputRef.current?.click()} className="rounded-full">
            <PlusCircle className="w-4 h-4 mr-2" /> Add image
          </Button>
          <div className="text-xs text-zinc-400">JPG, PNG, WebP • Max 10MB</div>
        </div>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the ad you want (tone, facts, CTA). We never invent facts."
          className="w-full p-4 min-h-24 rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-200 placeholder-zinc-500"
        />
        <Button
          onClick={handleSubmit}
          disabled={files.length === 0}
          className="rounded-2xl py-4 w-full bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white shadow-lg"
        >
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


