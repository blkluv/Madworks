"use client"

import { useApp } from "./app-context"

export function CreditsPill({ variant = "fixed" }: { variant?: "fixed" | "inline" }) {
  const { credits } = useApp()
  const containerClass = variant === "fixed" ? "fixed top-2 right-2 z-50" : "relative z-10"
  const sizeClass = variant === "inline" ? "h-12 px-4" : "px-4 py-2"
  return (
    <div className={containerClass}>
      <div className={`${sizeClass} inline-flex items-center rounded-full bg-zinc-950/80 backdrop-blur-md border border-zinc-800 text-zinc-100 text-base font-semibold shadow-lg whitespace-nowrap`}>
        {Number.isInteger(credits) ? credits : credits.toFixed(2)} credits left
      </div>
    </div>
  )
}


