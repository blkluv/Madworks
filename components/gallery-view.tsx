"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Filter, Search, Heart, Download, BookOpen } from "lucide-react"
import { useState } from "react"
import { useApp } from "./app-context"

const mockSavedTemplates = []

export function GalleryView() {
  const { gallery } = useApp()
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("All")

  const categories = ["All", "E-commerce", "Typography", "Design", "Health & Fitness", "Sustainability", "Technology"]

  const filteredSavedTemplates = mockSavedTemplates.filter((item) => {
    const matchesSearch = item.title.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = selectedCategory === "All" || item.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4">
          My Gallery
        </h1>
        <p className="text-gray-300 text-lg max-w-2xl mx-auto">Your processed images and generated ads.</p>
      </div>

      {/* Search and Filter Bar */}
      <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search saved templates..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-slate-700/80 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all duration-300 rounded-2xl"
            />
          </div>
          <div className="flex items-center gap-3">
            <Filter className="text-gray-400 w-5 h-5" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-4 py-3 bg-slate-700/80 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all duration-300 rounded-2xl"
            >
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {gallery.map((item) => (
          <Card
            key={item.id}
            className="bg-zinc-950/90 backdrop-blur overflow-hidden hover:bg-zinc-900 transition-all duration-500 rounded-2xl group cursor-pointer border border-zinc-900"
          >
            <div className="relative">
              <img src={item.thumbnail_url || "/placeholder.svg"} alt={item.title} className="w-full h-48 object-cover transition-all duration-500 group-hover:scale-105" />
              <div className="absolute top-3 right-3">
                <div className="bg-zinc-900/90 backdrop-blur p-2 rounded-full border border-zinc-800">
                  <Heart className="w-4 h-4 text-zinc-200" />
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-bold text-white text-lg leading-tight">{item.title}</h3>
                <div className="flex items-center gap-1 text-gray-400 text-sm">
                  <Download className="w-4 h-4" />
                  <span>{item.variants?.length || 0}</span>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {item.variants?.map((v) => (
                  <a key={`${v.format}-${v.w}x${v.h}`} href={v.url} download className="px-3 py-1 rounded-full bg-zinc-900 text-zinc-200 border border-zinc-800 text-xs">
                    {v.format.toUpperCase()} {v.w}×{v.h}
                  </a>
                ))}
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="text-center py-16">
        <div className="max-w-md mx-auto">
          <div className="p-6 bg-zinc-900 rounded-3xl w-fit mx-auto mb-6 border border-zinc-800"><BookOpen className="w-16 h-16 text-white" /></div>
          <h3 className="text-2xl font-bold text-white mb-4">No Saved Templates Yet</h3>
          <p className="text-gray-300 leading-relaxed mb-6">Process images to see them here with multi-format downloads.</p>
          <Button className="bg-white text-black rounded-2xl px-8 py-3 font-semibold shadow">Upload Now</Button>
        </div>
      </div>
    </div>
  )
}
