"use client"

import React from 'react'
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Heart, Download, Eye, TrendingUp, Crown, Zap, Code } from "lucide-react"
import { useState } from "react"

// Mock developer templates data
const mockDeveloperTemplates = [
  {
    id: 1,
    title: "Professional SaaS Landing",
    author: "Madworks Team",
    authorAvatar: "/woman-designer.png",
    uploadDate: "2024-01-20T08:30:00",
    likes: 542,
    downloads: 328,
    views: 2150,
    score: 98,
    category: "SaaS",
    thumbnail: "/modern-tech-product-ad.png",
    tags: ["professional", "saas", "conversion", "premium"],
    featured: true,
    isPremium: true,
  },
  {
    id: 2,
    title: "E-commerce Product Showcase",
    author: "Design Studio Pro",
    authorAvatar: "/creative-man.png",
    uploadDate: "2024-01-19T15:45:00",
    likes: 387,
    downloads: 195,
    views: 1580,
    score: 96,
    category: "E-commerce",
    thumbnail: "/food-delivery-ad.png",
    tags: ["ecommerce", "product", "showcase", "sales"],
    featured: false,
    isPremium: true,
  },
  {
    id: 3,
    title: "Luxury Brand Campaign",
    author: "Elite Creatives",
    authorAvatar: "/woman-photographer.png",
    uploadDate: "2024-01-18T12:20:00",
    likes: 656,
    downloads: 403,
    views: 2890,
    score: 99,
    category: "Luxury",
    thumbnail: "/tropical-getaway-ad.png",
    tags: ["luxury", "premium", "brand", "elegant"],
    featured: true,
    isPremium: true,
  },
]

// Mock AI-generated templates data
const mockAITemplates = [
  {
    id: 4,
    title: "AI-Generated Fashion Ad",
    author: "Madworks AI",
    authorAvatar: "/stylish-person.png",
    uploadDate: "2024-01-17T10:15:00",
    likes: 298,
    downloads: 167,
    views: 1220,
    score: 92,
    category: "Fashion",
    thumbnail: "/summer-sale-ad.png",
    tags: ["ai-generated", "fashion", "trendy", "automated"],
    featured: false,
    isPremium: true,
    isAIGenerated: true,
  },
  {
    id: 5,
    title: "AI Fitness Motivation",
    author: "Madworks AI",
    authorAvatar: "/diverse-fitness-trainer.png",
    uploadDate: "2024-01-16T14:30:00",
    likes: 424,
    downloads: 212,
    views: 1600,
    score: 94,
    category: "Health & Fitness",
    thumbnail: "/fitness-motivation-ad.png",
    tags: ["ai-generated", "fitness", "motivation", "health"],
    featured: false,
    isPremium: true,
    isAIGenerated: true,
  },
  {
    id: 6,
    title: "AI Eco-Friendly Campaign",
    author: "Madworks AI",
    authorAvatar: "/woman-environmentalist.png",
    uploadDate: "2024-01-15T09:45:00",
    likes: 367,
    downloads: 189,
    views: 1350,
    score: 93,
    category: "Sustainability",
    thumbnail: "/eco-friendly-products.png",
    tags: ["ai-generated", "eco", "sustainable", "green"],
    featured: false,
    isPremium: true,
    isAIGenerated: true,
  },
]

export function PremiumTemplatesView() {
  const [filters, setFilters] = useState({
    category: "all",
    templateType: "all", // 'developer', 'ai', 'all'
    sortBy: "popular", // 'popular', 'newest', 'downloads', 'rating'
    tags: [] as string[],
    searchQuery: ""
  })

  const categories = [
    { value: "all", label: "All Categories" },
    { value: "saas", label: "SaaS" },
    { value: "ecommerce", label: "E-commerce" },
    { value: "luxury", label: "Luxury" },
    { value: "fashion", label: "Fashion" },
    { value: "fitness", label: "Health & Fitness" },
    { value: "sustainability", label: "Sustainability" },
  ]

  const sortOptions = [
    { value: "popular", label: "Most Popular" },
    { value: "newest", label: "Newest First" },
    { value: "downloads", label: "Most Downloaded" },
    { value: "rating", label: "Highest Rated" },
  ]

  const availableTags = [
    { id: "featured", name: "Featured", color: "from-indigo-600 to-indigo-700" },
    { id: "trending", name: "Trending", color: "from-pink-600 to-rose-500" },
    { id: "new", name: "New", color: "from-emerald-500 to-teal-500" },
    { id: "free", name: "Free", color: "from-amber-500 to-orange-500" },
  ]

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffTime = Math.abs(now.getTime() - date.getTime())
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))

    if (diffDays === 1) return "1 day ago"
    if (diffDays < 7) return `${diffDays} days ago`
    if (diffDays < 30) return `${Math.ceil(diffDays / 7)} weeks ago`
    return `${Math.ceil(diffDays / 30)} months ago`
  }

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Crown className="w-8 h-8 text-yellow-400" />
          <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
            Premium Templates
          </h1>
        </div>
        <p className="text-gray-300 text-lg max-w-2xl mx-auto">
          Access exclusive developer-crafted templates and AI-generated designs for professional campaigns.
        </p>
      </div>

      {/* Premium Stats - Centered with max width */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-7xl mx-auto mb-8">
        <Card className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-indigo-600 to-indigo-700 rounded-2xl shadow-sm">
              <Code className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Developer Templates</p>
              <p className="text-white text-xl font-bold">847</p>
            </div>
          </div>
        </Card>
        <Card className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-sm">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">AI Templates</p>
              <p className="text-white text-xl font-bold">3,256</p>
            </div>
          </div>
        </Card>
        <Card className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-indigo-600 to-indigo-700 rounded-2xl shadow-sm">
              <Download className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Premium Downloads</p>
              <p className="text-white text-xl font-bold">45.8K</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Enhanced Filter Bar */}
      <Card className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 rounded-2xl shadow-lg mb-8">
        <div className="space-y-6">
          {/* Search Bar */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search templates..."
              value={filters.searchQuery}
              onChange={(e) => setFilters({...filters, searchQuery: e.target.value})}
              className="w-full px-4 py-3 bg-zinc-900/70 border border-zinc-800 text-zinc-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 transition-all duration-300 pl-12"
            />
            <svg
              className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-zinc-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-4">
              {/* Template Type Filter */}
              <div className="flex flex-col">
                <span className="text-sm text-zinc-400 mb-1">Template Type</span>
                <div className="flex items-center gap-2 bg-zinc-900/50 p-1 rounded-xl">
                  {[
                    { value: 'all', label: 'All' },
                    { value: 'developer', label: 'Developer' },
                    { value: 'ai', label: 'AI Generated' }
                  ].map((type) => (
                    <button
                      key={type.value}
                      onClick={() => setFilters({...filters, templateType: type.value as any})}
                      className={`px-4 py-2 text-sm rounded-lg transition-colors ${filters.templateType === type.value 
                        ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white' 
                        : 'text-zinc-300 hover:bg-zinc-800/50'}`}
                    >
                      {type.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Category Filter */}
              <div className="flex flex-col">
                <span className="text-sm text-zinc-400 mb-1">Category</span>
                <select
                  value={filters.category}
                  onChange={(e) => setFilters({...filters, category: e.target.value})}
                  className="px-4 py-2 bg-zinc-900/70 border border-zinc-800 text-zinc-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 transition-all duration-300"
                >
                  {categories.map((category) => (
                    <option key={category.value} value={category.value}>
                      {category.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sort By */}
              <div className="flex flex-col">
                <span className="text-sm text-zinc-400 mb-1">Sort By</span>
                <select
                  value={filters.sortBy}
                  onChange={(e) => setFilters({...filters, sortBy: e.target.value as any})}
                  className="px-4 py-2 bg-zinc-900/70 border border-zinc-800 text-zinc-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 transition-all duration-300"
                >
                  {sortOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Tags Filter */}
          <div className="space-y-2">
            <span className="text-sm text-zinc-400">Popular Tags</span>
            <div className="flex flex-wrap gap-2">
              {availableTags.map((tag) => (
                <button
                  key={tag.id}
                  onClick={() => {
                    const newTags = filters.tags.includes(tag.id)
                      ? filters.tags.filter(t => t !== tag.id)
                      : [...filters.tags, tag.id]
                    setFilters({...filters, tags: newTags})
                  }}
                  className={`px-3 py-1.5 text-sm rounded-full transition-all flex items-center gap-1.5 ${
                    filters.tags.includes(tag.id)
                      ? `bg-gradient-to-r ${tag.color} text-white`
                      : 'bg-zinc-900/50 text-zinc-300 hover:bg-zinc-800/70'
                  }`}
                >
                  {tag.name}
                  {filters.tags.includes(tag.id) && <span className="text-xs">×</span>}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Developer Created Templates */}
      <div className="space-y-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <Code className="w-6 h-6 text-indigo-400" />
          <h2 className="text-2xl font-bold text-white">Developer Created Templates</h2>
          <div className="px-3 py-1 bg-gradient-to-r from-indigo-600 to-indigo-700 text-white text-xs font-bold rounded-full">
            Premium
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockDeveloperTemplates.map((template) => (
            <Card
              key={template.id}
              className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 hover:bg-zinc-900/80 transition-all duration-500 hover:shadow-2xl hover:shadow-indigo-500/15 rounded-2xl relative overflow-hidden"
            >
              {template.featured && (
                <div className="absolute top-4 right-4 bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg z-10">
                  Featured
                </div>
              )}

              <div className="relative mb-4">
                <img
                  src={template.thumbnail || "/placeholder.svg"}
                  alt={template.title}
                  className="w-full h-48 object-cover rounded-xl shadow-lg"
                />
                <div className="absolute top-3 left-3 bg-zinc-900/90 backdrop-blur px-2 py-1 rounded-full text-xs font-bold text-emerald-400 border border-zinc-800">
                  {template.score}
                </div>
                <div className="absolute top-3 right-3 bg-gradient-to-r from-yellow-500 to-yellow-600 text-white px-2 py-1 rounded-full text-xs font-bold shadow-lg">
                  <Crown className="w-3 h-3 inline mr-1" />
                  PRO
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-bold text-white text-lg leading-tight">{template.title}</h3>

                <div className="flex items-center gap-3">
                  <img
                    src={template.authorAvatar || "/placeholder.svg"}
                    alt={template.author}
                    className="w-8 h-8 rounded-full object-cover"
                  />
                  <div>
                    <p className="text-gray-300 font-medium text-sm">{template.author}</p>
                    <p className="text-gray-500 text-xs">{formatDate(template.uploadDate)}</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {template.tags.slice(0, 3).map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-zinc-800 text-indigo-300 text-xs rounded-full border border-zinc-700/60">
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-between text-sm text-gray-400">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <Heart className="w-4 h-4 text-pink-400" />
                      <span>{template.likes}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Download className="w-4 h-4 text-indigo-400" />
                      <span>{template.downloads}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4 text-gray-400" />
                      <span>{template.views}</span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    className="flex-1 bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Use Template
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-zinc-900 text-gray-300 hover:bg-zinc-800 border border-zinc-800 rounded-2xl transition-all duration-300"
                  >
                    <Heart className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Most-Used AI-Generated Templates */}
      <div className="space-y-6 pt-8 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <Zap className="w-6 h-6 text-emerald-400" />
          <h2 className="text-2xl font-bold text-white">Most-Used AI-Generated Templates</h2>
          <div className="px-3 py-1 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white text-xs font-bold rounded-full">
            AI Powered
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockAITemplates.map((template) => (
            <Card
              key={template.id}
              className="bg-zinc-950/80 border border-zinc-900 backdrop-blur p-6 hover:bg-zinc-900/80 transition-all duration-500 hover:shadow-2xl hover:shadow-emerald-500/15 rounded-2xl relative overflow-hidden"
            >
              <div className="relative mb-4">
                <img
                  src={template.thumbnail || "/placeholder.svg"}
                  alt={template.title}
                  className="w-full h-48 object-cover rounded-xl shadow-lg"
                />
                <div className="absolute top-3 left-3 bg-zinc-900/90 backdrop-blur px-2 py-1 rounded-full text-xs font-bold text-emerald-400 border border-zinc-800">
                  {template.score}
                </div>
                <div className="absolute top-3 right-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white px-2 py-1 rounded-full text-xs font-bold shadow-lg">
                  <Zap className="w-3 h-3 inline mr-1" />
                  AI
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-bold text-white text-lg leading-tight">{template.title}</h3>

                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600 flex items-center justify-center">
                    <Zap className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <p className="text-gray-300 font-medium text-sm">{template.author}</p>
                    <p className="text-gray-500 text-xs">{formatDate(template.uploadDate)}</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {template.tags.slice(0, 3).map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-zinc-800 text-emerald-300 text-xs rounded-full border border-zinc-700/60">
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-between text-sm text-gray-400">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1">
                      <Heart className="w-4 h-4 text-pink-400" />
                      <span>{template.likes}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Download className="w-4 h-4 text-emerald-400" />
                      <span>{template.downloads}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4 text-gray-400" />
                      <span>{template.views}</span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    className="flex-1 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Use Template
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-zinc-900 text-gray-300 hover:bg-zinc-800 border border-zinc-800 rounded-2xl transition-all duration-300"
                  >
                    <Heart className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Load More */}
      <div className="text-center pt-8">
        <Button className="bg-gradient-to-r from-indigo-600 via-pink-600 to-orange-500 hover:from-indigo-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-8 py-3 shadow-lg hover:shadow-xl transition-all duration-300">
          Load More Templates
        </Button>
      </div>
    </div>
  )
}
