"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  MessageSquare,
  ImageIcon,
  Sparkles,
  Filter,
  Search,
  Calendar,
  Eye,
  Download,
  Heart,
  MoreVertical,
  Folder,
  Wand2,
} from "lucide-react"

type ProjectItem = {
  id: string
  type: "upload" | "conversation" | "generated"
  title: string
  date: string
  thumbnail: string
  description: string
  tags: string[]
  stats?: {
    views?: number
    downloads?: number
    likes?: number
  }
}

const mockProjects: ProjectItem[] = []

export function ProjectsView() {
  const [filter, setFilter] = useState<"all" | "upload" | "conversation" | "generated">("all")
  const [searchTerm, setSearchTerm] = useState("")

  const filteredProjects = mockProjects.filter((project) => {
    const matchesFilter = filter === "all" || project.type === filter
    const matchesSearch =
      project.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      project.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      project.tags.some((tag) => tag.toLowerCase().includes(searchTerm.toLowerCase()))
    return matchesFilter && matchesSearch
  })

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "upload":
        return <ImageIcon className="w-4 h-4" />
      case "conversation":
        return <MessageSquare className="w-4 h-4" />
      case "generated":
        return <Sparkles className="w-4 h-4" />
      default:
        return <Folder className="w-4 h-4" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case "upload":
        return "bg-blue-600 text-white"
      case "conversation":
        return "bg-purple-600 text-white"
      case "generated":
        return "bg-pink-600 text-white"
      default:
        return "bg-slate-600 text-white"
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-3 mb-4">
          <div className="p-4 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 rounded-2xl shadow-xl shadow-purple-600/25">
            <Folder className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
            My Projects
          </h1>
        </div>
        <p className="text-gray-300 text-lg max-w-2xl mx-auto">
          All your uploads, AI conversations, and generated content in one place
        </p>
      </div>

      {/* Controls */}
      <div className="bg-slate-800/90 backdrop-blur-sm rounded-3xl p-6 mb-8 shadow-xl shadow-purple-900/20">
        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search projects..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-slate-700 text-white placeholder-gray-400 rounded-2xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <div className="flex gap-2">
              {[
                { key: "all", label: "All", icon: Folder },
                { key: "upload", label: "Uploads", icon: ImageIcon },
                { key: "conversation", label: "Chats", icon: MessageSquare },
                { key: "generated", label: "AI Generated", icon: Wand2 },
              ].map(({ key, label, icon: Icon }) => (
                <Button
                  key={key}
                  onClick={() => setFilter(key as any)}
                  className={`rounded-2xl px-4 py-2 text-sm font-medium transition-all ${
                    filter === key
                      ? "bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 text-white shadow-lg"
                      : "bg-slate-700 text-gray-300 hover:bg-slate-600"
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {label}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Projects Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((project) => (
          <div
            key={project.id}
            className="group bg-slate-800/90 backdrop-blur-sm rounded-3xl overflow-hidden shadow-xl shadow-purple-900/20 hover:shadow-2xl hover:shadow-purple-500/30 transition-all duration-300 hover:-translate-y-1"
          >
            {/* Thumbnail */}
            <div className="relative h-48 overflow-hidden">
              <img
                src={project.thumbnail || "/placeholder.svg"}
                alt={project.title}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
              <div className="absolute top-3 left-3">
                <Badge className={`${getTypeColor(project.type)} rounded-full px-3 py-1 text-xs font-medium`}>
                  {getTypeIcon(project.type)}
                  <span className="ml-1 capitalize">{project.type}</span>
                </Badge>
              </div>
              <div className="absolute top-3 right-3">
                <Button className="w-8 h-8 bg-slate-700/90 backdrop-blur-sm text-gray-300 hover:bg-slate-600 rounded-full p-0">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Content */}
            <div className="p-6">
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-bold text-white text-lg leading-tight group-hover:text-purple-300 transition-colors">
                  {project.title}
                </h3>
              </div>

              <p className="text-gray-300 text-sm mb-4 line-clamp-2">{project.description}</p>

              {/* Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {project.tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="px-2 py-1 bg-slate-700 text-purple-300 text-xs rounded-full">
                    #{tag}
                  </span>
                ))}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between pt-4 border-t border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Calendar className="w-3 h-3" />
                  {new Date(project.date).toLocaleDateString()}
                </div>

                {project.stats && (
                  <div className="flex items-center gap-4 text-xs text-gray-400">
                    {project.stats.views && (
                      <div className="flex items-center gap-1">
                        <Eye className="w-3 h-3" />
                        {project.stats.views}
                      </div>
                    )}
                    {project.stats.downloads && (
                      <div className="flex items-center gap-1">
                        <Download className="w-3 h-3" />
                        {project.stats.downloads}
                      </div>
                    )}
                    {project.stats.likes && (
                      <div className="flex items-center gap-1">
                        <Heart className="w-3 h-3" />
                        {project.stats.likes}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredProjects.length === 0 && (
        <div className="text-center py-16">
          <div className="max-w-md mx-auto">
            <div className="p-6 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 rounded-3xl w-fit mx-auto mb-6 shadow-2xl shadow-purple-500/30">
              <Folder className="w-16 h-16 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Start Your First Project</h3>
            <p className="text-gray-300 leading-relaxed mb-6">
              Upload your first advertisement, chat with our AI assistant, or generate new variations to get started.
            </p>
            <Button className="bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 hover:from-purple-700 hover:via-pink-700 hover:to-orange-600 text-white rounded-2xl px-8 py-3 font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
              <Wand2 className="w-5 h-5 mr-2" />
              Create New Project
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
