"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FolderOpen, BookOpen, Users, Crown, Check, Sparkles, TrendingUp, Zap, Shield } from "lucide-react"
import { HomeChatBox } from "./home-chat-box"

export function HomeView() {
  return (
    <div className="max-w-6xl mx-auto space-y-8 sm:space-y-12">
      {/* Hero Section (no big box) */}
      <div className="text-center space-y-4 sm:space-y-6">
        <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
          <span className="bg-gradient-to-r from-indigo-400 via-pink-500 to-purple-500 bg-clip-text text-transparent">Transform</span> Your Ads with AI
        </h1>
        <p className="text-gray-300 text-base sm:text-lg max-w-3xl mx-auto px-2">
          Upload an image and describe your goal. We’ll generate crisp, on‑brand ads across sizes and formats.
        </p>
      </div>

      {/* Chat handoff to Uploads */}
      <div className="max-w-3xl mx-auto">
        <HomeChatBox />
      </div>

      {/* Info Section */}
      <div className="max-w-6xl mx-auto pt-8 sm:pt-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
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
