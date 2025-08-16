"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Check, Crown, Sparkles } from "lucide-react"
import { useRouter } from "next/navigation"
import HomePage from "../page"

export default function UpgradePage() {
  const router = useRouter()
  return (
    <div className="min-h-screen bg-black text-zinc-100">
      <div className="container mx-auto px-4 py-6">
        {/* Reuse the same header/menu from the homepage by rendering HomePage header-only would be complex; we keep global header from layout */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-4 bg-zinc-900/60 border border-zinc-800 rounded-2xl backdrop-blur">
              <Crown className="w-8 h-8 text-zinc-200" />
            </div>
            <h1 className="text-4xl font-bold">Upgrade your plan</h1>
          </div>
          <p className="text-zinc-400 max-w-2xl mx-auto">
            Simple, transparent pricing. Pick a plan that fits your workflow. You can cancel anytime.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <Card className="bg-zinc-950/80 border border-zinc-900 rounded-3xl p-8 backdrop-blur flex flex-col">
            <div className="mb-6">
              <h3 className="text-xl font-bold mb-1">Bascc</h3>
              <div className="text-3xl font-bold">$0</div>
              <div className="text-zinc-400">per month</div>
            </div>
            <ul className="space-y-3 text-zinc-300 mb-8">
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> 5 uploads / month</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Basic analysis</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Community templates</li>
            </ul>
            <Button className="rounded-2xl py-3 w-full mt-auto" onClick={() => router.push("/")}>Stay Free</Button>
          </Card>

          <Card className="bg-zinc-900/70 border border-zinc-800 rounded-3xl p-8 backdrop-blur flex flex-col relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white text-xs">Most Popular</div>
            <div className="mb-6">
              <h3 className="text-xl font-bold mb-1 flex items-center gap-2"><Sparkles className="w-4 h-4" /> Pro</h3>
              <div className="text-3xl font-bold">$19</div>
              <div className="text-zinc-400">per month</div>
            </div>
            <ul className="space-y-3 text-zinc-300 mb-8">
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> 25 monthly credits</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Pay‑per‑generation beyond 25</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Advanced AI analysis</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Priority rendering</li>
            </ul>
            <Button className="rounded-2xl py-3 w-full mt-auto bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white">Upgrade to Pro</Button>
          </Card>

          <Card className="bg-zinc-950/80 border border-zinc-900 rounded-3xl p-8 backdrop-blur flex flex-col">
            <div className="mb-6">
              <h3 className="text-xl font-bold mb-1 flex items-center gap-2"><Crown className="w-4 h-4" /> Enterprise</h3>
              <div className="text-3xl font-bold">$99</div>
              <div className="text-zinc-400">per month</div>
            </div>
            <ul className="space-y-3 text-zinc-300 mb-8">
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Unlimited credits</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Team collaboration</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Custom brand kits</li>
              <li className="flex items-center gap-2"><Check className="w-4 h-4 text-emerald-500" /> Dedicated support</li>
            </ul>
            <Button className="rounded-2xl py-3 w-full mt-auto">Contact Sales</Button>
          </Card>
        </div>
      </div>
    </div>
  )
}


