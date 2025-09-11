"use client"

import { Suspense, useState, useEffect } from "react"
import { useSearchParams } from "next/navigation"
import { HomeView } from "@/components/home-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { EditorView } from "@/components/editor-view"
import { ChatView } from "@/components/chat-view"

type ViewType = "home" | "studio" | "gallery" | "premium" | "chat" | "refine" | "projects"

function HomeContent() {
  return (
    <div className={`h-full min-h-0 text-zinc-100 relative overflow-visible`}>
      <Suspense fallback={
        <div className="flex items-center justify-center h-full">
          <div className="animate-pulse">Loading...</div>
        </div>
      }>
        <HomeContentInner />
      </Suspense>
    </div>
  )
}

function HomeContentInner() {
  const searchParams = useSearchParams()
  const [currentView, setCurrentView] = useState<ViewType>(() => {
    const v = (searchParams.get("view") as ViewType) || "home"
    return v
  })

  // Sync view from ?view= query param
  useEffect(() => {
    const v = (searchParams.get("view") as ViewType) || "home"
    setCurrentView(v)
  }, [searchParams])

  return (
    <div className="relative z-10 h-full min-h-0 flex flex-col">
      {currentView === "chat" ? (
        // Full-bleed chat dashboard
        <ChatView />
      ) : (
        <div className="w-full max-w-none px-4 md:px-8 pt-8 pb-8">
          {currentView === "home" && <HomeView />}
          {(currentView === "studio" || currentView === "refine" || currentView === "projects") && <EditorView />}
          {currentView === "gallery" && <PremiumTemplatesView />}
        </div>
      )}
    </div>
  )
}

export default HomeContent

