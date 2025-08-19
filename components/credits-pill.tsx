"use client"

import { useApp } from "./app-context"

export function CreditsPill({ variant = "fixed" }: { variant?: "fixed" | "inline" }) {
  const { credits } = useApp()
  const containerClass = variant === "fixed" ? "fixed top-2 right-2 z-50" : "relative z-10"
  const pillClass =
    variant === "inline"
      ? "h-8 px-3 inline-flex items-center rounded-full bg-zinc-900/40 backdrop-blur border border-zinc-800/70 text-zinc-300 text-sm shadow"
      : "px-4 py-2 inline-flex items-center rounded-full bg-zinc-950/80 backdrop-blur-md border border-zinc-800 text-zinc-100 text-base font-semibold shadow-lg"
  return (
    <div className={containerClass}>
      <div className={`${pillClass} whitespace-nowrap`}>
        {Number.isInteger(credits) ? credits : credits.toFixed(2)} credits left
      </div>
    </div>
  )
}


