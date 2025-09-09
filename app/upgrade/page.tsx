"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Check, Crown, Sparkles } from "lucide-react"
import { useRouter } from "next/navigation"

export default function UpgradePage() {
  const router = useRouter()
  return (
    <div className="min-h-screen text-zinc-100">
      <div className="container mx-auto px-4 pt-6 pb-10">
        {/* Reuse the same header/menu from the homepage by rendering HomePage header-only would be complex; we keep global header from layout */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-3 mb-3">
            <div className="p-3 bg-zinc-900/60 border border-zinc-800 rounded-xl backdrop-blur">
              <Crown className="w-7 h-7 text-zinc-200" />
            </div>
            <h1 className="text-2xl md:text-3xl font-bold">Upgrade your plan</h1>
          </div>
          <p className="text-zinc-400 text-sm max-w-xl mx-auto">
            Simple, transparent pricing. Pick a plan that fits your workflow. You can cancel anytime.
          </p>
        </div>

        <div className="grid md:grid-cols-1 gap-5 max-w-xl mx-auto">
          <Card className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-6 backdrop-blur flex flex-col relative">
            <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2.5 py-0.5 rounded-full bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white text-[10px] tracking-wide">Beta Access</div>
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-1 flex items-center gap-2"><Sparkles className="w-4 h-4" /> Starter Kit</h3>
              <div className="text-2xl font-bold">$5</div>
              <div className="text-zinc-400 text-xs">one-time beta testing kit</div>
            </div>
            <p className="text-sm text-zinc-400 mb-5">This Starter Kit is our early‑access, beta testing kit — a lightweight way to try modern ad generation and help us shape what ships next.</p>
            <ul className="space-y-2.5 text-zinc-300 text-sm mb-6">
              <li className="flex items-center gap-2"><Check className="w-3.5 h-3.5 text-emerald-500" /> Early access to new features</li>
              <li className="flex items-center gap-2"><Check className="w-3.5 h-3.5 text-emerald-500" /> A small batch of generation credits</li>
              <li className="flex items-center gap-2"><Check className="w-3.5 h-3.5 text-emerald-500" /> Dedicated beta feedback channel</li>
            </ul>
            <Button className="rounded-xl py-2.5 text-sm w-full mt-auto bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white">Get Starter Kit</Button>
          </Card>
        </div>
      </div>
    </div>
  )
}


