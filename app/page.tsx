"use client"

import { useState, useEffect } from "react"
import { HomeView } from "@/components/home-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { ProjectsView } from "@/components/projects-view"
import { ChatView } from "@/components/chat-view"
import { useSearchParams } from "next/navigation"

type ViewType = "home" | "projects" | "gallery" | "premium" | "chat"

export default function HomePage() {
  const [currentView, setCurrentView] = useState<ViewType>("home")
  const searchParams = useSearchParams()

  // Sync view from ?view= query param
  useEffect(() => {
    const v = (searchParams.get("view") as ViewType) || "home"
    setCurrentView(v)
  }, [searchParams])

  // keep page focused on view rendering; header handles navigation

  return (
    <div className={`min-h-screen text-zinc-100 relative overflow-x-hidden ${currentView === "chat" ? "overflow-hidden" : "overflow-y-auto"}`}>
      {/* Main Content */}
      <div className={`relative z-10 ${currentView === "chat" ? "pt-4" : "pt-8"}`}>
        <div className={`container mx-auto px-4 ${currentView === "chat" ? "pb-4" : "pb-8"}`}>
          {currentView === "home" && <HomeView />}
          {currentView === "projects" && <ProjectsView />}
          {currentView === "gallery" && <PremiumTemplatesView />}
          {currentView === "chat" && <ChatView />}
        </div>
      </div>
    </div>
  )
}

