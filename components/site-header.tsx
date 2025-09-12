"use client"

import { Button } from "@/components/ui/button"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle, Menu, X } from "lucide-react"
import { useState, useEffect, useRef } from "react"
import { useSession } from "next-auth/react"
import { useSearchParams, useRouter } from "next/navigation"
import Image from "next/image"

type ViewType = "home" | "studio" | "gallery" | "premium" | "chat"

export function SiteHeader({ currentView, onNavChange }: { currentView?: ViewType; onNavChange?: (v: ViewType) => void }) {
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { data: session, status } = useSession()
  const isAuthed = status === 'authenticated'
  const dropdownRef = useRef<HTMLDivElement>(null)
  const searchParams = useSearchParams()
  const router = useRouter()

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
  // Lock body scroll when mobile menu is open
  useEffect(() => {
    try {
      if (mobileMenuOpen) {
        document.body.style.overflow = 'hidden'
      } else {
        document.body.style.overflow = ''
      }
    } catch {}
    return () => {
      try { document.body.style.overflow = '' } catch {}
    }
  }, [mobileMenuOpen])
  const goto = (view: ViewType) => {
    try {
      if (onNavChange) onNavChange(view)
      else router.push(`/?view=${view}`)
    } catch {
      // Fallback to hard nav only if router fails
      if (typeof window !== 'undefined') window.location.href = `/?view=${view}`
    }
  }
  const selectedView: ViewType = (currentView ?? ((searchParams.get("view") as ViewType) || "home"))
  return (
    <div className="sticky top-0 z-50" style={{ paddingTop: 'env(safe-area-inset-top, 0px)' }}>
      <div className="w-full relative overflow-visible bg-transparent">
        <div className="mx-auto max-w-7xl px-4 py-2 sm:py-3 relative z-10">
          <div className="flex items-center justify-between gap-2 sm:gap-3 rounded-2xl md:rounded-full px-3 sm:px-4 py-2 bg-[linear-gradient(90deg,rgba(88,101,242,0.28)_0%,rgba(236,72,153,0.20)_50%,rgba(147,51,234,0.28)_100%)] backdrop-blur-md ring-1 ring-white/20 shadow-[0_6px_24px_rgba(0,0,0,0.35)]">
          <div
            role="button"
            onClick={() => goto("home")}
            className="flex items-center gap-3 cursor-pointer group justify-self-start"
            aria-label="Go to Home"
            title="Madworks AI - Home"
          >
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <Image
                src="/mwlg2.png"
                alt="Madworks logo"
                width={150}
                height={50}
                className="h-9 sm:h-10 md:h-12 w-auto object-contain"
                priority
              />
              <span className="inline text-sm sm:text-base md:text-xl lg:text-3xl font-bold text-white select-none truncate max-w-[55vw] sm:max-w-[50vw] md:max-w-[40vw] lg:max-w-none">Madworks AI</span>
            </div>
          </div>

          {/* Desktop / Tablet nav */}
          <div className="hidden lg:flex flex-1 min-w-0 justify-center -mx-2 px-2">
            <nav className="flex items-center gap-2 whitespace-nowrap">
              <div className="flex items-center gap-2">
                <Button
                  data-nav="home"
                  onClick={() => goto("home")}
                  variant="outline"
                  size="lg"
                  className={`h-9 md:h-11 lg:h-12 px-3 md:px-5 lg:px-6 text-sm md:text-base group relative isolate overflow-visible rounded-xl font-semibold border backdrop-blur-sm shadow-md shadow-black/30 ring-1 ring-white/10 ${selectedView === 'home' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  {selectedView === 'home' && (
                    <span className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#f59e0b_0deg,#6366f1_120deg,#ec4899_240deg,#f59e0b_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0" />
                  )}
                  <Home className="w-4 h-4 mr-2" /> Home
                </Button>
                <Button
                  data-nav="chat"
                  onClick={() => goto("chat")}
                  variant="outline"
                  size="lg"
                  className={`h-9 md:h-11 lg:h-12 px-3 md:px-5 lg:px-6 text-sm md:text-base group relative isolate overflow-visible rounded-xl font-semibold border backdrop-blur-sm shadow-md shadow-black/30 ring-1 ring-white/10 ${selectedView === 'chat' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  {selectedView === 'chat' && (
                    <span className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#f59e0b_0deg,#6366f1_120deg,#ec4899_240deg,#f59e0b_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0" />
                  )}
                  <BookOpen className="w-4 h-4 mr-2" /> Create
                </Button>
                <Button
                  data-nav="studio"
                  onClick={() => goto("studio")}
                  variant="outline"
                  size="lg"
                  className={`h-9 md:h-11 lg:h-12 px-3 md:px-5 lg:px-6 text-sm md:text-base group relative isolate overflow-visible rounded-xl font-semibold border backdrop-blur-sm shadow-md shadow-black/30 ring-1 ring-white/10 ${selectedView === 'studio' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  {selectedView === 'studio' && (
                    <span className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#f59e0b_0deg,#6366f1_120deg,#ec4899_240deg,#f59e0b_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0" />
                  )}
                  <Folder className="w-4 h-4 mr-2" /> Studio
                </Button>
                <Button
                  data-nav="gallery"
                  onClick={() => goto("gallery")}
                  variant="outline"
                  size="lg"
                  className={`h-9 md:h-11 lg:h-12 px-3 md:px-5 lg:px-6 text-sm md:text-base group relative isolate overflow-visible rounded-xl font-semibold border backdrop-blur-sm shadow-md shadow-black/30 ring-1 ring-white/10 ${selectedView === 'gallery' ? 'bg-zinc-900/70 text-white border-zinc-700 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]' : 'bg-zinc-900/50 text-zinc-200 border-zinc-800 hover:bg-zinc-900/70'}`}
                >
                  {selectedView === 'gallery' && (
                    <span className="pointer-events-none absolute -inset-[2px] rounded-2xl bg-[conic-gradient(at_0%_0%,#f59e0b_0deg,#6366f1_120deg,#ec4899_240deg,#f59e0b_360deg)] opacity-10 group-hover:opacity-20 blur-sm transition-opacity z-0" />
                  )}
                  <BookOpen className="w-4 h-4 mr-2" /> Premium Templates
                </Button>
              </div>
            </nav>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 justify-self-end flex-shrink-0">
            {/* Right cluster: Upgrade, User avatar */}
            <Button
              onClick={() => (window.location.href = "/upgrade")}
              className="hidden lg:inline-flex h-9 md:h-11 lg:h-12 px-4 md:px-5 rounded-xl bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-600 hover:from-yellow-500 hover:via-amber-500 hover:to-amber-700 text-black text-sm md:text-base shadow-lg shadow-yellow-500/25 backdrop-blur-sm ring-1 ring-white/10"
            >
              <Crown className="w-4 h-4 mr-2" /> Upgrade
            </Button>
            {/* Mobile menu button */}
            <Button
              size="icon"
              className="lg:hidden h-9 w-9 p-0 rounded-xl bg-zinc-900/60 hover:bg-zinc-900/70 text-white border border-zinc-700"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open menu"
            >
              <Menu className="w-5 h-5" />
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
                className="h-9 w-9 sm:h-10 sm:w-10 md:h-12 md:w-12 p-0 bg-zinc-900/60 hover:bg-zinc-900/70 text-white rounded-full border border-zinc-700 flex items-center justify-center overflow-visible backdrop-blur-sm shadow-md shadow-black/30 ring-1 ring-white/10"
              >
                {isAuthed && session?.user?.image ? (
                  // Show Google user avatar in a fixed-size circular frame
                  <div className="w-8 h-8 sm:w-9 sm:h-9 md:w-10 md:h-10 rounded-full overflow-hidden shadow">
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

      {/* Mobile full-screen menu overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm" onClick={() => setMobileMenuOpen(false)}>
          {/* subtle gradient wash */}
          <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(80%_60%_at_50%_0%,rgba(99,102,241,0.12),transparent_60%)]" />
          <div
            className="absolute inset-0"
            style={{ paddingTop: 'env(safe-area-inset-top, 12px)', paddingBottom: 'env(safe-area-inset-bottom, 12px)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col h-full">
              <div className="flex items-center justify-between px-4 py-3">
                <div className="flex items-center gap-2">
                  <Image src="/mwlg2.png" alt="Madworks logo" width={120} height={40} className="h-8 w-auto" />
                  <span className="text-xl font-bold">Madworks AI</span>
                </div>
                <Button size="icon" className="h-9 w-9 p-0 rounded-xl bg-zinc-900/60 border border-zinc-700" aria-label="Close menu" onClick={() => setMobileMenuOpen(false)}>
                  <X className="w-5 h-5" />
                </Button>
              </div>

              <nav className="mt-1 px-4 space-y-2 overflow-y-auto pb-8">
                <Button onClick={() => { setMobileMenuOpen(false); goto('home') }} className={`w-full h-12 rounded-xl justify-start bg-zinc-900/60 border-zinc-800 ${selectedView==='home' ? 'ring-1 ring-white/20' : ''}`}>
                  <Home className="w-5 h-5 mr-3" /> Home
                </Button>
                <Button onClick={() => { setMobileMenuOpen(false); goto('chat') }} className={`w-full h-12 rounded-xl justify-start bg-zinc-900/60 border-zinc-800 ${selectedView==='chat' ? 'ring-1 ring-white/20' : ''}`}>
                  <BookOpen className="w-5 h-5 mr-3" /> Create
                </Button>
                <Button onClick={() => { setMobileMenuOpen(false); goto('studio') }} className={`w-full h-12 rounded-xl justify-start bg-zinc-900/60 border-zinc-800 ${selectedView==='studio' ? 'ring-1 ring-white/20' : ''}`}>
                  <Folder className="w-5 h-5 mr-3" /> Studio
                </Button>
                <Button onClick={() => { setMobileMenuOpen(false); goto('gallery') }} className={`w-full h-12 rounded-xl justify-start bg-zinc-900/60 border-zinc-800 ${selectedView==='gallery' ? 'ring-1 ring-white/20' : ''}`}>
                  <BookOpen className="w-5 h-5 mr-3" /> Premium Templates
                </Button>

                <div className="pt-4">
                  <Button onClick={() => { setMobileMenuOpen(false); window.location.href='/upgrade' }} className="w-full h-12 rounded-xl bg-gradient-to-r from-yellow-400 via-yellow-500 to-amber-600 text-black">
                    <Crown className="w-5 h-5 mr-3" /> Upgrade
                  </Button>
                </div>
              </nav>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
