"use client"

import { Button } from "@/components/ui/button"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"
import { useState } from "react"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export function SiteHeader({ currentView, onNavChange }: { currentView?: ViewType; onNavChange?: (v: ViewType) => void }) {
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)
  const goto = (view: ViewType) => {
    if (onNavChange) onNavChange(view)
    else window.location.href = `/?view=${view}`
  }
  return (
    <div className="sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25">
              <div className="w-6 h-6 flex items-center justify-center">
                <span className="text-white font-bold text-xl">M</span>
              </div>
            </div>
            <h1 className="text-2xl font-bold text-white">Madworks AI</h1>
          </div>

          <div className="absolute left-1/2 transform -translate-x-1/2">
            <nav className="flex justify-center">
              <div className="flex items-center gap-2">
                <Button
                  data-nav="home"
                  onClick={() => goto("home")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold ${currentView === 'home' ? 'bg-black/70 text-white' : ''}`}
                >
                  <Home className="w-4 h-4 mr-2" /> Home
                </Button>
                <Button
                  data-nav="chat"
                  onClick={() => goto("chat")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold ${currentView === 'chat' ? 'bg-black/70 text-white' : ''}`}
                >
                  <BookOpen className="w-4 h-4 mr-2" /> Create
                </Button>
                <Button
                  data-nav="projects"
                  onClick={() => goto("projects")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold ${currentView === 'projects' ? 'bg-black/70 text-white' : ''}`}
                >
                  <Folder className="w-4 h-4 mr-2" /> Projects
                </Button>
                <Button
                  data-nav="gallery"
                  onClick={() => goto("gallery")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold ${currentView === 'gallery' ? 'bg-black/70 text-white' : ''}`}
                >
                  <BookOpen className="w-4 h-4 mr-2" /> Premium Templates
                </Button>
              </div>
            </nav>
          </div>

          <div className="flex items-center gap-2">
            <Button
              onClick={() => (window.location.href = "/upgrade")}
              className="h-10 px-4 rounded-xl bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-600 hover:from-yellow-500 hover:via-amber-500 hover:to-amber-700 text-black shadow-lg shadow-yellow-500/25"
            >
              <Crown className="w-4 h-4 mr-2" /> Upgrade
            </Button>
            <div className="relative">
              <Button
                onClick={() => setShowAccountDropdown((s) => !s)}
                className="h-12 w-12 bg-black/40 hover:bg-black/60 text-white rounded-full shadow border border-zinc-800 flex items-center justify-center"
              >
                <User className="w-4 h-4" />
              </Button>
              {showAccountDropdown && (
                <div className="absolute top-full mt-2 left-1/2 -translate-x-1/2 w-56 bg-zinc-950/95 backdrop-blur rounded-xl shadow border border-zinc-800 py-1 px-1">
                  <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                    <UserCircle className="w-4 h-4 mr-2" /> View Profile
                  </Button>
                  <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                    <User className="w-4 h-4 mr-2" /> Generative Preferences
                  </Button>
                  <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                    <HelpCircle className="w-4 h-4 mr-2" /> FAQ
                  </Button>
                  <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                    <Settings className="w-4 h-4 mr-2" /> Settings
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
