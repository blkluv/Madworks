"use client"

import { useState } from "react"
import { HomeView } from "@/components/home-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { ProjectsView } from "@/components/projects-view"
import { Button } from "@/components/ui/button"
import { useApp } from "@/components/app-context"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"
import { SiteHeader } from "@/components/site-header"
import { ChatView } from "@/components/chat-view"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export default function HomePage() {
  const { credits } = useApp()
  const [currentView, setCurrentView] = useState<ViewType>("home")
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)

  const closeAllDropdowns = () => {
    setShowAccountDropdown(false)
  }

  const handleViewChange = (view: ViewType) => {
    closeAllDropdowns()
    setCurrentView(view)
  }

  const handleUpgradeClick = () => {
    closeAllDropdowns()
    setCurrentView("home")
    // Scroll to pricing section if on home page
    setTimeout(() => {
      const pricingSection = document.getElementById("pricing-section")
      if (pricingSection) {
        pricingSection.scrollIntoView({ behavior: "smooth" })
      }
    }, 100)
  }

  return (
    <div className="min-h-screen bg-black text-zinc-100 relative overflow-x-hidden overflow-y-auto">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Subtle marble texture overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-900/20 via-pink-900/10 to-orange-900/20 opacity-30"></div>

        {/* Flowing organic shapes - progressively more towards bottom */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-br from-indigo-600/5 to-pink-600/5 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 right-1/3 w-80 h-80 bg-gradient-to-br from-pink-600/5 to-orange-600/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 left-1/3 w-[500px] h-[500px] bg-gradient-to-br from-orange-600/8 to-indigo-600/8 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/6 right-1/4 w-[400px] h-[400px] bg-gradient-to-br from-indigo-600/8 to-pink-600/8 rounded-full blur-3xl"></div>

        {/* Subtle veining patterns */}
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-indigo-500/2 to-transparent"></div>
        <div className="absolute inset-0 bg-gradient-to-tl from-transparent via-pink-500/2 to-transparent"></div>
      </div>

      <SiteHeader currentView={currentView} onNavChange={handleViewChange} />

      {/* Main Content */}
      <div className="relative z-10 pt-8">
        <div className="container mx-auto px-4 pb-8">
          {currentView === "home" && <HomeView />}
          {currentView === "projects" && <ProjectsView />}
          {currentView === "gallery" && <PremiumTemplatesView />}
          {currentView === "chat" && <ChatView />}
        </div>
      </div>
    </div>
  )
}

