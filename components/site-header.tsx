"use client"

import { Button } from "@/components/ui/button"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"
import { useState, useEffect, useRef } from "react"
import { useSession } from "next-auth/react"
import { useSearchParams } from "next/navigation"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export function SiteHeader({ currentView, onNavChange }: { currentView?: ViewType; onNavChange?: (v: ViewType) => void }) {
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)
  const { data: session, status } = useSession()
  const isAuthed = status === 'authenticated'
  const dropdownRef = useRef<HTMLDivElement>(null)
  const searchParams = useSearchParams()

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowAccountDropdown(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])
  const goto = (view: ViewType) => {
    if (onNavChange) onNavChange(view)
    else window.location.href = `/?view=${view}`
  }
  const selectedView: ViewType = (currentView ?? ((searchParams.get("view") as ViewType) || "home"))
  return (
    <div className="sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="grid grid-cols-[auto_1fr_auto] items-center gap-4">
          <div
            role="button"
            onClick={() => goto("home")}
            className="flex items-center gap-3 cursor-pointer group justify-self-start"
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
                  className={`rounded-xl px-6 py-3 font-semibold border ${selectedView === 'home' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  <Home className="w-4 h-4 mr-2" /> Home
                </Button>
                <Button
                  data-nav="chat"
                  onClick={() => goto("chat")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold border ${selectedView === 'chat' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  <BookOpen className="w-4 h-4 mr-2" /> Create
                </Button>
                <Button
                  data-nav="projects"
                  onClick={() => goto("projects")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold border ${selectedView === 'projects' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  <Folder className="w-4 h-4 mr-2" /> Projects
                </Button>
                <Button
                  data-nav="gallery"
                  onClick={() => goto("gallery")}
                  variant="outline"
                  size="lg"
                  className={`rounded-xl px-6 py-3 font-semibold border ${selectedView === 'gallery' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  <BookOpen className="w-4 h-4 mr-2" /> Premium Templates
                </Button>
              </div>
            </nav>
          </div>

          <div className="flex items-center gap-3 justify-self-end">
            {/* Right cluster: Upgrade, User avatar */}
            <Button
              onClick={() => (window.location.href = "/upgrade")}
              className="h-12 px-5 rounded-xl bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-600 hover:from-yellow-500 hover:via-amber-500 hover:to-amber-700 text-black shadow-lg shadow-yellow-500/25"
            >
              <Crown className="w-4 h-4 mr-2" /> Upgrade
            </Button>
            <div className="relative" ref={dropdownRef}>
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
                  <div className="w-10 h-10 rounded-full overflow-hidden shadow">
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
              <div 
                className={`absolute top-full mt-2 right-0 w-56 bg-zinc-950/95 backdrop-blur rounded-xl shadow-lg border border-zinc-800 overflow-hidden transition-all duration-200 ease-out origin-top ${
                  showAccountDropdown 
                    ? 'opacity-100 scale-100 translate-y-0 pointer-events-auto' 
                    : 'opacity-0 scale-95 -translate-y-2 pointer-events-none'
                }`}
                onClick={(e) => e.stopPropagation()}
              >
                {isAuthed ? (
                  <>
                    <div className="p-4 border-b border-zinc-800">
                      <p className="font-medium text-white">{session?.user?.name || 'User'}</p>
                      <p className="text-xs text-zinc-400 truncate">{session?.user?.email}</p>
                    </div>
                    <div className="py-1">
                      <Button 
                        onClick={() => window.location.href = "/profile"} 
                        className="w-full text-left px-4 py-2.5 text-sm text-zinc-200 hover:bg-zinc-900/80 bg-transparent rounded-none justify-start"
                      >
                        <UserCircle className="w-4 h-4 mr-3 text-zinc-400" /> 
                        View Profile
                      </Button>
                      <Button 
                        onClick={() => window.location.href = "/settings"}
                        className="w-full text-left px-4 py-2.5 text-sm text-zinc-200 hover:bg-zinc-900/80 bg-transparent rounded-none justify-start"
                      >
                        <Settings className="w-4 h-4 mr-3 text-zinc-400" /> 
                        Settings
                      </Button>
                      <Button 
                        className="w-full text-left px-4 py-2.5 text-sm text-zinc-200 hover:bg-zinc-900/80 bg-transparent rounded-none justify-start"
                      >
                        <HelpCircle className="w-4 h-4 mr-3 text-zinc-400" /> 
                        Help & Support
                      </Button>
                    </div>
                    <div className="p-2 border-t border-zinc-800">
                      <Button 
                        variant="outline"
                        className="w-full text-sm bg-transparent hover:bg-zinc-900/50 border-zinc-800 text-zinc-200 hover:text-white"
                        onClick={() => {
                          // Handle sign out
                          window.location.href = '/api/auth/signout';
                        }}
                      >
                        Sign Out
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="p-4">
                    <p className="text-sm text-zinc-300 mb-3">Sign in to access your account</p>
                    <Button 
                      onClick={() => window.location.href = '/login'}
                      className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white"
                    >
                      Sign In
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
