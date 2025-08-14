"use client"

import { useApp } from "./app-context"

export function CreditsPill() {
  const { credits } = useApp()
  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="px-3 py-1.5 rounded-full bg-zinc-950/70 backdrop-blur border border-zinc-800 text-zinc-200 text-sm font-medium shadow">
        {credits} credits left
      </div>
    </div>
  )
}


