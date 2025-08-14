"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Filter, Search, Heart, Download, BookOpen } from "lucide-react"
import { useState } from "react"

const mockSavedTemplates = []

export function GalleryView() {
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
        <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-400 via-pink-400 to-orange-400 bg-clip-text text-transparent mb-4">
          My Gallery
        </h1>
        <p className="text-gray-300 text-lg max-w-2xl mx-auto">
          Your personal collection of saved templates from the community.
        </p>
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
        {filteredSavedTemplates.map((template) => (
          <Card
            key={template.id}
            className="bg-slate-800/90 backdrop-blur-sm overflow-hidden hover:bg-slate-700/90 transition-all duration-500 hover:shadow-2xl hover:shadow-indigo-500/15 rounded-2xl group cursor-pointer"
          >
            <div className="relative">
              <img
                src={template.thumbnail || "/placeholder.svg"}
                alt={template.title}
                className="w-full h-48 object-cover transition-all duration-500 group-hover:scale-105"
              />
              <div className="absolute top-3 right-3">
                <div className="bg-slate-700/90 backdrop-blur-sm p-2 rounded-full">
                  <Heart className="w-4 h-4 text-pink-400 fill-current" />
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-bold text-white text-lg leading-tight">{template.title}</h3>
                <div className="flex items-center gap-1 text-gray-400 text-sm">
                  <Download className="w-4 h-4" />
                  <span>{template.downloads}</span>
                </div>
              </div>
              <div className="flex items-center gap-3 mb-4">
                <img
                  src={template.authorAvatar || "/placeholder.svg"}
                  alt={template.author}
                  className="w-8 h-8 rounded-full object-cover"
                />
                <div>
                  <p className="text-gray-300 font-medium text-sm">{template.author}</p>
                  <p className="text-gray-500 text-xs">{template.category}</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mb-4">
                {template.tags.map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-slate-700 text-indigo-300 rounded-full text-xs font-medium">
                    {tag}
                  </span>
                ))}
              </div>
              <div className="flex items-center justify-between text-sm text-gray-400">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <Heart className="w-4 h-4 text-pink-400" />
                    <span>{template.likes}</span>
                  </div>
                </div>
                <Button
                  size="sm"
                  className="bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-4 py-2 shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  Use Template
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="text-center py-16">
        <div className="max-w-md mx-auto">
          <div className="p-6 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-3xl w-fit mx-auto mb-6 shadow-2xl shadow-indigo-500/30">
            <BookOpen className="w-16 h-16 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">No Saved Templates Yet</h3>
          <p className="text-gray-300 leading-relaxed mb-6">
            Discover and save amazing templates from the Premium Templates section to build your personal collection.
          </p>
          <Button className="bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-8 py-3 font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
            <Search className="w-5 h-5 mr-2" />
            Browse Premium Templates
          </Button>
        </div>
      </div>
    </div>
  )
}
