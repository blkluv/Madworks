"use client"

import { useState } from "react"
import { HomeView } from "@/components/home-view"
import { UploadInterface } from "@/components/upload-interface"
import { GalleryView } from "@/components/gallery-view"
import { PremiumTemplatesView } from "@/components/premium-templates-view"
import { ProjectsView } from "@/components/projects-view"
import { Button } from "@/components/ui/button"
import { ImageIcon, Folder, BookOpen, Home, Crown, User, Settings, UserCircle, HelpCircle } from "lucide-react"

type ViewType = "home" | "upload" | "projects" | "gallery" | "premium"

export default function HomePage() {
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-blue-900 text-white relative overflow-x-hidden overflow-y-auto scrollbar-thin scrollbar-track-slate-800 scrollbar-thumb-indigo-600 hover:scrollbar-thumb-indigo-500">
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
      <div className="sticky top-0 z-50 bg-gradient-to-br from-slate-900/95 via-indigo-900/95 to-blue-900/95 backdrop-blur-sm border-b border-slate-700/50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            {/* Logo and brand name */}
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25">
                <div className="w-6 h-6 flex items-center justify-center">
                  <span className="text-white font-bold text-xl">M</span>
                </div>
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
                Madworks AI
              </h1>
            </div>

            <div className="absolute left-1/2 transform -translate-x-1/2">
              {/* Navigation Pills */}
              <nav className="flex justify-center">
                <div className="bg-slate-800/80 backdrop-blur-sm rounded-full p-2 shadow-2xl shadow-indigo-900/20">
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={() => handleViewChange("home")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "home"
                          ? "bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white shadow-lg shadow-indigo-600/30"
                          : "bg-transparent text-gray-300 hover:bg-slate-700/50"
                      }`}
                    >
                      <Home className="w-4 h-4 mr-2" />
                      Home
                    </Button>
                    <Button
                      onClick={() => handleViewChange("upload")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "upload"
                          ? "bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white shadow-lg shadow-indigo-600/30"
                          : "bg-transparent text-gray-300 hover:bg-slate-700/50"
                      }`}
                    >
                      <ImageIcon className="w-4 h-4 mr-2" />
                      Upload
                    </Button>
                    <Button
                      onClick={() => handleViewChange("projects")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "projects"
                          ? "bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white shadow-lg shadow-indigo-600/30"
                          : "bg-transparent text-gray-300 hover:bg-slate-700/50"
                      }`}
                    >
                      <Folder className="w-4 h-4 mr-2" />
                      Projects
                    </Button>
                    <Button
                      onClick={() => handleViewChange("gallery")}
                      className={`rounded-full px-6 py-3 font-semibold transition-all duration-300 ${
                        currentView === "gallery"
                          ? "bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white shadow-lg shadow-indigo-600/30"
                          : "bg-transparent text-gray-300 hover:bg-slate-700/50"
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
                className="h-12 px-4 bg-gradient-to-r from-yellow-500 to-amber-600 hover:from-yellow-400 hover:to-amber-500 text-black font-semibold rounded-full shadow-lg shadow-yellow-500/25 transition-all duration-300 hover:scale-105"
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
                  className="h-12 w-12 bg-slate-800/80 backdrop-blur-sm hover:bg-slate-700/80 text-white rounded-full shadow-lg shadow-slate-900/25 transition-all duration-300 hover:scale-105 border border-slate-600/30 flex items-center justify-center"
                >
                  <User className="w-4 h-4" />
                </Button>

                {showAccountDropdown && (
                  <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 w-56 bg-slate-800/95 backdrop-blur-sm rounded-xl shadow-2xl shadow-indigo-900/20 border border-slate-700/50 py-1 px-1">
                    <Button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-slate-700/50 bg-transparent rounded-lg transition-colors duration-200 justify-start">
                      <UserCircle className="w-4 h-4 mr-2" />
                      View Profile
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-slate-700/50 bg-transparent rounded-lg transition-colors duration-200 justify-start">
                      <User className="w-4 h-4 mr-2" />
                      Generative Preferences
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-slate-700/50 bg-transparent rounded-lg transition-colors duration-200 justify-start">
                      <HelpCircle className="w-4 h-4 mr-2" />
                      FAQ
                    </Button>
                    <Button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-slate-700/50 bg-transparent rounded-lg transition-colors duration-200 justify-start">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </Button>
                    <Button
                      onClick={handleUpgradeClick}
                      className="w-full text-left px-3 py-2 text-yellow-400 hover:bg-slate-700/50 bg-transparent rounded-lg transition-colors duration-200 justify-start"
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
