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

      {/* Info Section */}
      <div className="max-w-6xl mx-auto pt-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="bg-zinc-950/80 border border-zinc-900 rounded-2xl p-6">
            <div className="flex items-start gap-3">
              <Zap className="w-5 h-5 text-emerald-400 mt-1" />
              <div>
                <h3 className="text-white font-semibold">AI Copy & Layout</h3>
                <p className="text-zinc-400 text-sm">Smart copy, emphasis, and layout optimized for legibility and conversions.</p>
              </div>
            </div>
          </Card>

          <Card className="bg-zinc-950/80 border border-zinc-900 rounded-2xl p-6">
            <div className="flex items-start gap-3">
              <TrendingUp className="w-5 h-5 text-pink-400 mt-1" />
              <div>
                <h3 className="text-white font-semibold">Multiple Sizes</h3>
                <p className="text-zinc-400 text-sm">Generate consistent variants for square, vertical, and horizontal placements.</p>
              </div>
            </div>
          </Card>

          <Card className="bg-zinc-950/80 border border-zinc-900 rounded-2xl p-6">
            <div className="flex items-start gap-3">
              <Shield className="w-5 h-5 text-indigo-400 mt-1" />
              <div>
                <h3 className="text-white font-semibold">On‑brand Controls</h3>
                <p className="text-zinc-400 text-sm">Override text color, choose panel side, and fine‑tune layouts with preferences.</p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
