"use client"

import { Button } from "@/components/ui/button"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"
import { useState } from "react"
import { useSession } from "next-auth/react"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export function SiteHeader({ currentView, onNavChange }: { currentView?: ViewType; onNavChange?: (v: ViewType) => void }) {
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)
  const { data: session, status } = useSession()
  const isAuthed = status === 'authenticated'
  const goto = (view: ViewType) => {
    if (onNavChange) onNavChange(view)
    else window.location.href = `/?view=${view}`
  }
  return (
    <div className="sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="grid grid-cols-[auto_1fr_auto] items-center gap-4">
          <div
            role="button"
            onClick={() => goto("home")}
            className="flex items-center gap-3 cursor-pointer group"
            aria-label="Go to Home"
            title="Madworks AI - Home"
          >
            <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25 ring-0 group-hover:ring-2 group-hover:ring-white/40 transition">
              <div className="w-6 h-6 flex items-center justify-center">
                <span className="text-white font-bold text-xl">M</span>
              </div>
            </div>
            <h1 className="text-2xl font-bold text-white">Madworks AI</h1>
          </div>

          <div className="justify-self-center">
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
                size="icon"
                onClick={() => {
                  if (!isAuthed) {
                    window.location.href = "/login"
                  } else {
                    setShowAccountDropdown((s) => !s)
                  }
                }}
                className="h-12 w-12 p-0 bg-zinc-900/70 hover:bg-zinc-900 text-white rounded-full shadow border border-zinc-700 flex items-center justify-center overflow-visible"
              >
                {isAuthed && session?.user?.image ? (
                  // Show Google user avatar in a fixed-size circular frame
                  <div className="w-10 h-10 rounded-full overflow-hidden ring-2 ring-white/90 shadow shrink-0">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={session.user.image}
                      alt={session.user.name ?? "User"}
                      className="w-full h-full object-cover block"
                    />
                  </div>
                ) : (
                  <User className="w-5 h-5" />
                )}
              </Button>
              {showAccountDropdown && (
                <div className="absolute top-full mt-2 left-1/2 -translate-x-1/2 w-56 bg-zinc-950/95 backdrop-blur rounded-xl shadow border border-zinc-800 py-1 px-1">
                  <Button onClick={() => (window.location.href = "/profile")} className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                    <UserCircle className="w-4 h-4 mr-2" /> View Profile
                  </Button>
                  <Button onClick={() => (window.location.href = "/preferences")} className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
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
