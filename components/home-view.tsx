"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FolderOpen, BookOpen, Users, Crown, Check, Sparkles, TrendingUp, Zap, Shield } from "lucide-react"
import { HomeChatBox } from "./home-chat-box"

export function HomeView() {
  return (
    <div className="max-w-6xl mx-auto space-y-12">
      {/* Hero Section (no big box) */}
      <div className="text-center space-y-6">
        <div className="inline-flex items-center justify-center p-4 rounded-3xl bg-zinc-900/50 border border-zinc-800 backdrop-blur">
          <Sparkles className="h-10 w-10 text-zinc-300" />
        </div>
        <h1 className="text-4xl font-bold text-white tracking-tight">
          <span className="chromatic-text">Transform</span> Your Ads with AI
        </h1>
        <p className="text-gray-300 text-lg max-w-3xl mx-auto">
          Upload an image and describe your goal. We’ll generate crisp, on‑brand ads across sizes and formats.
        </p>
      </div>

      {/* Chat handoff to Uploads */}
      <div className="max-w-3xl mx-auto">
        <HomeChatBox />
      </div>
    </div>
  )
}
