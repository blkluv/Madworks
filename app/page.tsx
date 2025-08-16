"use client"

import { useState, useEffect } from "react"
import { HomeView } from "@/components/home-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { ProjectsView } from "@/components/projects-view"
import { Button } from "@/components/ui/button"
import { useApp } from "@/components/app-context"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"
import { ChatView } from "@/components/chat-view"
import { useSearchParams } from "next/navigation"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export default function HomePage() {
  const { credits } = useApp()
  const [currentView, setCurrentView] = useState<ViewType>("home")
  const [showAccountDropdown, setShowAccountDropdown] = useState(false)
  const searchParams = useSearchParams()

  // Sync view from ?view= query param
  useEffect(() => {
    const v = (searchParams.get("view") as ViewType) || "home"
    setCurrentView(v)
  }, [searchParams])

  const closeAllDropdowns = () => {
    setShowAccountDropdown(false)
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
    <div className="min-h-screen text-zinc-100 relative overflow-x-hidden overflow-y-auto">
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

