"use client"

import { useState } from "react"
import { HomeView } from "@/components/home-view"
import { UploadInterface } from "@/components/upload-interface"
import { GalleryView } from "@/components/gallery-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { ProjectsView } from "@/components/projects-view"
import { Button } from "@/components/ui/button"
import { useApp } from "@/components/app-context"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"

type ViewType = "home" | "upload" | "projects" | "gallery" | "premium"

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
      <div className="credits-pill">
        <div className="px-4 py-2 rounded-full bg-zinc-900/90 border border-zinc-800 text-zinc-200 text-sm font-medium shadow-lg">
          Credits: {credits}
        </div>
      </div>
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

      {/* Sticky Header */}
      <div className="sticky top-0 z-50 bg-black/85 backdrop-blur border-b border-zinc-800">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            {/* Logo and brand name */}
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25">
                <div className="w-6 h-6 flex items-center justify-center">
                  <span className="text-white font-bold text-xl">M</span>
                </div>
              </div>
              <h1 className="text-2xl font-bold text-white">Madworks AI</h1>
            </div>

            <div className="absolute left-1/2 transform -translate-x-1/2">
              {/* Navigation Pills */}
              <nav className="flex justify-center">
                <div className="bg-zinc-900/80 backdrop-blur rounded-full p-2 shadow-lg border border-zinc-800">
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={() => handleViewChange("home")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "home"
                          ? "bg-white text-black"
                          : "bg-transparent text-zinc-300 hover:bg-zinc-900"
                      }`}
                    >
                      <Home className="w-4 h-4 mr-2" />
                      Home
                    </Button>
                    <Button
                      onClick={() => handleViewChange("upload")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "upload"
                          ? "bg-white text-black"
                          : "bg-transparent text-zinc-300 hover:bg-zinc-900"
                      }`}
                    >
                      <ImageIcon className="w-4 h-4 mr-2" />
                      Upload
                    </Button>
                    <Button
                      onClick={() => handleViewChange("projects")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "projects"
                          ? "bg-white text-black"
                          : "bg-transparent text-zinc-300 hover:bg-zinc-900"
                      }`}
                    >
                      <Folder className="w-4 h-4 mr-2" />
                      Projects
                    </Button>
                    <Button
                      onClick={() => handleViewChange("gallery")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "gallery"
                          ? "bg-white text-black"
                          : "bg-transparent text-zinc-300 hover:bg-zinc-900"
                      }`}
                    >
                      <BookOpen className="w-4 h-4 mr-2" />
                      Gallery
                    </Button>
                    <Button
                      onClick={() => handleViewChange("premium")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "premium"
                          ? "bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white shadow-lg shadow-indigo-600/30"
                          : "bg-transparent text-gray-300 hover:bg-slate-700/50"
                      }`}
                    >
                      <Crown className="w-4 h-4 mr-2" />
                      Premium Templates
                    </Button>
                  </div>
                </div>
              </nav>
            </div>

            {/* User Controls */}
            <div className="flex items-center gap-2">
              <Button
                onClick={handleUpgradeClick}
                className="h-12 px-4 bg-white text-black font-semibold rounded-full shadow-md hover:shadow-lg transition-all"
              >
                <Crown className="w-4 h-4 mr-2" />
                Upgrade
              </Button>

              <div className="relative">
                <Button
                  onClick={() => {
                    closeAllDropdowns()
                    setShowAccountDropdown(!showAccountDropdown)
                  }}
                  className="h-12 w-12 bg-zinc-900 hover:bg-zinc-800 text-white rounded-full shadow border border-zinc-800 flex items-center justify-center"
                >
                  <User className="w-4 h-4" />
                </Button>

                {showAccountDropdown && (
                  <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 w-56 bg-zinc-950/95 backdrop-blur rounded-xl shadow border border-zinc-800 py-1 px-1">
                    <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                      <UserCircle className="w-4 h-4 mr-2" />
                      View Profile
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                      <User className="w-4 h-4 mr-2" />
                      Generative Preferences
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                      <HelpCircle className="w-4 h-4 mr-2" />
                      FAQ
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-zinc-300 hover:bg-zinc-900 bg-transparent rounded-lg justify-start">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </Button>
                    <Button
                      onClick={handleUpgradeClick}
                      className="w-full text-left px-3 py-2 text-white bg-black/40 hover:bg-black/60 rounded-lg justify-start"
                    >
                      <Crown className="w-4 h-4 mr-2" />
                      Upgrade
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 pt-8">
        <div className="container mx-auto px-4 pb-8">
          {currentView === "home" && <HomeView />}
          {currentView === "upload" && <UploadInterface />}
          {currentView === "projects" && <ProjectsView />}
          {currentView === "gallery" && <GalleryView />}
          {currentView === "premium" && <PremiumTemplatesView />}
        </div>
      </div>
    </div>
  )
}
