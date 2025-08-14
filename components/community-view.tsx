"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Heart, Download, Eye, Users, TrendingUp, Star } from "lucide-react"
import { useState } from "react"

// Mock community templates data
const mockCommunityTemplates = [
  {
    id: 1,
    title: "Minimalist Tech Product Launch",
    author: "Sarah Chen",
    authorAvatar: "/woman-designer.png",
    uploadDate: "2024-01-20T08:30:00",
    likes: 342,
    downloads: 128,
    views: 1250,
    score: 94,
    category: "Tech",
    thumbnail: "/modern-tech-product-ad.png",
    tags: ["minimal", "tech", "clean", "modern"],
    featured: true,
  },
  {
    id: 2,
    title: "Vibrant Food Delivery Campaign",
    author: "Marcus Rodriguez",
    authorAvatar: "/creative-man.png",
    uploadDate: "2024-01-19T15:45:00",
    likes: 287,
    downloads: 95,
    views: 980,
    score: 91,
    category: "Food & Beverage",
    thumbnail: "/food-delivery-ad.png",
    tags: ["colorful", "food", "delivery", "appetizing"],
    featured: false,
  },
  {
    id: 3,
    title: "Luxury Travel Experience",
    author: "Emma Thompson",
    authorAvatar: "/woman-photographer.png",
    uploadDate: "2024-01-18T12:20:00",
    likes: 456,
    downloads: 203,
    views: 1890,
    score: 96,
    category: "Travel",
    thumbnail: "/tropical-getaway-ad.png",
    tags: ["luxury", "travel", "tropical", "premium"],
    featured: true,
  },
  {
    id: 4,
    title: "Summer Fashion Collection",
    author: "Alex Kim",
    authorAvatar: "/stylish-person.png",
    uploadDate: "2024-01-17T10:15:00",
    likes: 198,
    downloads: 67,
    views: 720,
    score: 88,
    category: "Fashion",
    thumbnail: "/summer-sale-ad.png",
    tags: ["fashion", "summer", "trendy", "sale"],
    featured: false,
  },
  {
    id: 5,
    title: "Fitness Motivation Campaign",
    author: "Jordan Williams",
    authorAvatar: "/diverse-fitness-trainer.png",
    uploadDate: "2024-01-16T14:30:00",
    likes: 324,
    downloads: 112,
    views: 1100,
    score: 89,
    category: "Health & Fitness",
    thumbnail: "/fitness-motivation-ad.png",
    tags: ["fitness", "motivation", "health", "energy"],
    featured: false,
  },
  {
    id: 6,
    title: "Eco-Friendly Product Line",
    author: "Maya Patel",
    authorAvatar: "/woman-environmentalist.png",
    uploadDate: "2024-01-15T09:45:00",
    likes: 267,
    downloads: 89,
    views: 850,
    score: 92,
    category: "Sustainability",
    thumbnail: "/eco-friendly-products.png",
    tags: ["eco", "sustainable", "green", "natural"],
    featured: false,
  },
]

export function CommunityView() {
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [sortBy, setSortBy] = useState("popular")

  const categories = [
    { value: "all", label: "All Categories" },
    { value: "tech", label: "Tech" },
    { value: "food", label: "Food & Beverage" },
    { value: "travel", label: "Travel" },
    { value: "fashion", label: "Fashion" },
    { value: "fitness", label: "Health & Fitness" },
    { value: "sustainability", label: "Sustainability" },
  ]

  const sortOptions = [
    { value: "popular", label: "Most Popular" },
    { value: "recent", label: "Most Recent" },
    { value: "downloads", label: "Most Downloaded" },
    { value: "score", label: "Highest Score" },
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

  const featuredTemplates = mockCommunityTemplates.filter((template) => template.featured)
  const regularTemplates = mockCommunityTemplates.filter((template) => !template.featured)

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent mb-4">
          Community Templates
        </h1>
        <p className="text-gray-300 text-lg max-w-2xl mx-auto">
          Discover and download popular advertisement templates created by our creative community.
        </p>
      </div>

      {/* Community Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl shadow-sm">
              <Users className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Active Creators</p>
              <p className="text-white text-xl font-bold">2,847</p>
            </div>
          </div>
        </Card>
        <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-sm">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Templates</p>
              <p className="text-white text-xl font-bold">12,456</p>
            </div>
          </div>
        </Card>
        <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl shadow-sm">
              <Download className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Downloads</p>
              <p className="text-white text-xl font-bold">89.2K</p>
            </div>
          </div>
        </Card>
        <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-pink-500 to-pink-600 rounded-2xl shadow-sm">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-gray-400 text-sm font-medium">Total Likes</p>
              <p className="text-white text-xl font-bold">156K</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filter Bar */}
      <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="flex items-center gap-3">
              <span className="text-white font-semibold">Category:</span>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-4 py-2 bg-slate-700/80 text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all duration-300 rounded-2xl"
              >
                {categories.map((category) => (
                  <option key={category.value} value={category.value}>
                    {category.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-white font-semibold">Sort by:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-4 py-2 bg-slate-700/80 text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all duration-300 rounded-2xl"
              >
                {sortOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <Button className="bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 hover:from-purple-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-6 py-2 shadow-lg hover:shadow-xl transition-all duration-300">
            Upload Template
          </Button>
        </div>
      </Card>

      {/* Featured Templates */}
      {featuredTemplates.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <Star className="w-6 h-6 text-orange-400" />
            <h2 className="text-2xl font-bold text-white">Featured Templates</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {featuredTemplates.map((template) => (
              <Card
                key={template.id}
                className="bg-slate-800/90 backdrop-blur-sm p-6 hover:bg-slate-700/90 transition-all duration-500 hover:shadow-2xl hover:shadow-purple-500/15 rounded-2xl relative overflow-hidden"
              >
                <div className="absolute top-4 right-4 bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg z-10">
                  Featured
                </div>

                <div className="relative mb-4">
                  <img
                    src={template.thumbnail || "/placeholder.svg"}
                    alt={template.title}
                    className="w-full h-48 object-cover rounded-xl shadow-lg"
                  />
                  <div className="absolute top-3 left-3 bg-slate-700/90 backdrop-blur-sm px-2 py-1 rounded-full text-xs font-bold text-emerald-400">
                    {template.score}
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
                      <span key={tag} className="px-2 py-1 bg-slate-700 text-purple-300 text-xs rounded-full">
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
                        <Download className="w-4 h-4 text-blue-400" />
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
                      className="flex-1 bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 hover:from-purple-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Use Template
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="bg-slate-700 text-gray-300 hover:bg-slate-600 rounded-2xl transition-all duration-300"
                    >
                      <Heart className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Regular Templates */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-white">Popular Templates</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {regularTemplates.map((template) => (
            <Card
              key={template.id}
              className="bg-slate-800/90 backdrop-blur-sm p-6 hover:bg-slate-700/90 transition-all duration-500 hover:shadow-2xl hover:shadow-purple-500/15 rounded-2xl"
            >
              <div className="relative mb-4">
                <img
                  src={template.thumbnail || "/placeholder.svg"}
                  alt={template.title}
                  className="w-full h-48 object-cover rounded-xl shadow-lg"
                />
                <div className="absolute top-3 left-3 bg-slate-700/90 backdrop-blur-sm px-2 py-1 rounded-full text-xs font-bold text-emerald-400">
                  {template.score}
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
                    <span key={tag} className="px-2 py-1 bg-slate-700 text-purple-300 text-xs rounded-full">
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
                      <Download className="w-4 h-4 text-blue-400" />
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
                    className="flex-1 bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 hover:from-purple-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Use Template
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-slate-700 text-gray-300 hover:bg-slate-600 rounded-2xl transition-all duration-300"
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
        <Button className="bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 hover:from-purple-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-8 py-3 shadow-lg hover:shadow-xl transition-all duration-300">
          Load More Templates
        </Button>
      </div>
    </div>
  )
}
