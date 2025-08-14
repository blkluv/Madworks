"use client"

import type React from "react"
import { useState, useCallback, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, ImageIcon, Sparkles, Trash2, RefreshCw, TrendingUp, Eye, Zap, Send, PlusCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { useApp } from "./app-context"

interface UploadedFile {
  file: File
  preview: string
  analysis?: {
    suggestions: string[]
    score: number
    improvements: string[]
  }
}

export function UploadInterface() {
  const { decrementCredit, addToGallery, currentProject } = useApp()
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analyzingIndex, setAnalyzingIndex] = useState<number | null>(null)
  const [chatMessage, setChatMessage] = useState("")
  const [variationCount, setVariationCount] = useState(3)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files).filter((file) => file.type.startsWith("image/"))
    processFiles(files)
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    processFiles(files)
  }, [])

  const processFiles = (files: File[]) => {
    files.forEach((file) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const preview = e.target?.result as string
        setUploadedFiles((prev) => [...prev, { file, preview }])
      }
      reader.readAsDataURL(file)
    })
  }

  const analyzeAd = async (index: number) => {
    setIsAnalyzing(true)
    setAnalyzingIndex(index)

    // Send to backend pipeline (stub)
    const form = new FormData()
    form.append("file", uploadedFiles[index].file)
    form.append("template_id", "story-bold-plate-v3")
    const res = await fetch("http://localhost:3001/v1/jobs", { method: "POST", body: form as any }).catch(() => null)
    await new Promise((resolve) => setTimeout(resolve, 1500))

    const mockAnalysis = {
      suggestions: [
        "Consider using a stronger call-to-action button with contrasting colors",
        "The text hierarchy could be improved with better font sizing",
        "Try positioning the main element higher in the composition for better visibility",
        "The color palette could be more vibrant to increase engagement rates",
        "Add more whitespace around key elements to improve readability",
      ],
      score: Math.floor(Math.random() * 25) + 75,
      improvements: [
        "Increase CTA button size by 25% for better mobile visibility",
        "Use complementary colors to create stronger visual hierarchy",
        "Implement the rule of thirds for better composition balance",
        "A/B test different headline variations for higher conversion",
        "Add subtle shadows to make elements pop from the background",
      ],
    }

    setUploadedFiles((prev) => prev.map((file, i) => (i === index ? { ...file, analysis: mockAnalysis } : file)))
    // Add to gallery with variants stub
    addToGallery({
      id: `${Date.now()}-${index}`,
      title: uploadedFiles[index].file.name,
      thumbnail_url: uploadedFiles[index].preview,
      createdAt: Date.now(),
      variants: [
        { format: "png", w: 1080, h: 1350, url: "/placeholder.svg" },
        { format: "png", w: 1080, h: 1080, url: "/placeholder.svg" },
        { format: "jpg", w: 1920, h: 1080, url: "/placeholder.svg" },
        { format: "pdf", w: 2550, h: 3300, url: "/placeholder.svg" },
        { format: "svg", w: 1080, h: 1080, url: "/placeholder.svg" },
      ],
    })
    decrementCredit(1)
    setIsAnalyzing(false)
    setAnalyzingIndex(null)
  }

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const reanalyze = (index: number) => {
    setUploadedFiles((prev) => prev.map((file, i) => (i === index ? { ...file, analysis: undefined } : file)))
  }

  const handleChatSubmit = () => {
    if (chatMessage.trim()) {
      console.log("Chat message:", chatMessage, "Variations:", variationCount)
      setChatMessage("")
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <Card className="bg-zinc-950/90 backdrop-blur overflow-hidden relative shadow-2xl rounded-3xl border border-zinc-900">
        <div className="relative p-8">
          <div
            className={cn(
              "relative p-16 border-2 border-dashed rounded-3xl transition-all duration-500 ease-out mx-4 my-4",
              "bg-zinc-900 backdrop-blur",
              "shadow-inner",
              isDragOver
                ? "border-zinc-400 scale-[1.01] shadow-2xl"
                : "border-zinc-800 hover:border-zinc-700 hover:shadow-xl",
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="text-center relative z-10 space-y-8">
              {/* Upload Icon and Title */}
              <div>
                <div
                  className={cn(
                    "mx-auto mb-6 p-6 rounded-3xl transition-all duration-500 shadow-2xl relative overflow-hidden w-fit bg-zinc-900 border border-zinc-800",
                  )}
                >
                  <Upload className={cn("h-12 w-12 transition-all duration-500 relative z-10", "text-zinc-300")} />
                </div>

                <h3 className="text-2xl font-bold text-white mb-3 tracking-tight">Drop your images here</h3>
                <p className="text-gray-300 text-base mb-6 leading-relaxed font-medium max-w-lg mx-auto">
                  Upload assets, then describe what you want. We’ll generate on-brand ads across sizes and formats.
                </p>

                <div className="p-3 rounded-2xl bg-zinc-900 border border-zinc-800 mb-6 max-w-md mx-auto text-zinc-300 text-sm">
                  Supported: JPG, PNG, WebP • Max 10MB
                </div>
              </div>

              {/* File Input */}
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileSelect} accept="image/*" multiple />
          <Button onClick={() => fileInputRef.current?.click()} className="rounded-full">
                <PlusCircle className="w-4 h-4 mr-2" /> Add images
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Chat-like instruction composer */}
      <Card className="bg-zinc-950/90 backdrop-blur p-6 rounded-3xl border border-zinc-900">
        <div className="flex flex-col md:flex-row gap-3 items-start md:items-center">
          <div className="flex-1 w-full">
            <textarea
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              placeholder="Describe what you want on the generated ads (e.g., upbeat headline, price $3.49, CTA Try it now)."
              className="w-full min-h-16 p-4 rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-200 placeholder-zinc-500"
            />
            <div className="text-xs text-zinc-400 mt-2">We never invent facts. Your text will be trimmed to fit templates.</div>
          </div>
          <Button
            onClick={() => (uploadedFiles[0] ? analyzeAd(0) : fileInputRef.current?.click())}
            disabled={!uploadedFiles[0] || isAnalyzing}
            className="rounded-full px-6 py-3 whitespace-nowrap"
          >
            <Send className="w-4 h-4 mr-2" /> Generate ads
          </Button>
        </div>
      </Card>

      {uploadedFiles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl shadow-sm">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-gray-400 text-sm font-medium">Total Uploads</p>
                <p className="text-white text-xl font-bold">{uploadedFiles.length}</p>
              </div>
            </div>
          </Card>
          <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-sm">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-gray-400 text-sm font-medium">Analyzed</p>
                <p className="text-white text-xl font-bold">{uploadedFiles.filter((f) => f.analysis).length}</p>
              </div>
            </div>
          </Card>
          <Card className="bg-slate-700/80 backdrop-blur-sm p-6 rounded-2xl shadow-lg">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-indigo-500 to-pink-500 rounded-2xl shadow-sm">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-gray-400 text-sm font-medium">Avg Score</p>
                <p className="text-white text-xl font-bold">
                  {uploadedFiles.filter((f) => f.analysis).length > 0
                    ? Math.round(
                        uploadedFiles.filter((f) => f.analysis).reduce((acc, f) => acc + (f.analysis?.score || 0), 0) /
                          uploadedFiles.filter((f) => f.analysis).length,
                      )
                    : 0}
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div className="grid gap-8">
          {uploadedFiles.map((uploadedFile, index) => (
            <Card
              key={index}
              className="bg-slate-800/90 backdrop-blur-sm p-8 hover:bg-slate-700/90 transition-all duration-500 hover:shadow-2xl hover:shadow-indigo-500/15 rounded-3xl overflow-hidden relative"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-slate-800/30 via-indigo-900/20 to-indigo-900/30 opacity-60"></div>

              <div className="grid lg:grid-cols-2 gap-8 relative z-10">
                {/* Image Preview */}
                <div className="space-y-6">
                  <div className="relative group">
                    <div className="p-4 bg-gradient-to-br from-slate-700 via-indigo-800 to-indigo-800 rounded-3xl shadow-2xl">
                      <div className="p-2 bg-gradient-to-br from-slate-600 to-indigo-700 rounded-2xl shadow-inner">
                        <img
                          src={uploadedFile.preview || "/placeholder.svg"}
                          alt={`Gallery piece ${index + 1}`}
                          className="w-full h-80 object-cover rounded-xl shadow-lg transition-all duration-500 group-hover:scale-[1.02] group-hover:shadow-2xl"
                        />
                      </div>
                    </div>

                    <div className="absolute top-6 right-6 flex gap-3">
                      <Button
                        onClick={() => removeFile(index)}
                        size="sm"
                        variant="destructive"
                        className="bg-slate-700/90 hover:bg-red-600/90 text-red-400 hover:text-white rounded-2xl backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                      {uploadedFile.analysis && (
                        <Button
                          onClick={() => reanalyze(index)}
                          size="sm"
                          className="bg-slate-700/90 hover:bg-indigo-600/90 text-indigo-400 hover:text-white rounded-2xl backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-slate-700/80 via-indigo-800/60 to-slate-700/80 p-6 rounded-2xl shadow-lg backdrop-blur-sm">
                    <div className="grid grid-cols-2 gap-6 text-sm">
                      <div>
                        <p className="text-indigo-300 mb-2 font-semibold uppercase tracking-wide text-xs">
                          Artwork Title
                        </p>
                        <p className="text-white font-bold truncate text-base">{uploadedFile.file.name}</p>
                      </div>
                      <div>
                        <p className="text-indigo-300 mb-2 font-semibold uppercase tracking-wide text-xs">File Size</p>
                        <p className="text-white font-bold text-base">
                          {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Analysis Section */}
                <div className="space-y-6">
                  {!uploadedFile.analysis ? (
                    <div className="text-center py-12">
                      <div className="mb-6">
                        <div
                          className={cn(
                            "mx-auto p-4 rounded-2xl transition-all duration-300 shadow-lg w-fit",
                            analyzingIndex === index
                              ? "bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 animate-pulse"
                              : "bg-gradient-to-br from-slate-700 to-indigo-700",
                          )}
                        >
                          <Sparkles
                            className={cn(
                              "h-10 w-10 transition-colors duration-300",
                              analyzingIndex === index ? "text-white" : "text-indigo-300",
                            )}
                          />
                        </div>
                      </div>
                      <h4 className="text-xl font-bold text-white mb-3">Ready for AI Analysis</h4>
                      <p className="text-gray-300 mb-6 max-w-sm mx-auto">
                        Our AI will analyze your design and provide actionable insights to improve performance.
                      </p>
                      <Button onClick={() => analyzeAd(index)} disabled={isAnalyzing} className={cn("rounded-2xl px-8 py-3 font-semibold shadow transition-all", isAnalyzing ? "opacity-50 cursor-not-allowed" : "hover:shadow-lg") }>
                        {analyzingIndex === index ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-3"></div>
                            Analyzing Design...
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-5 h-5 mr-3" />
                            Analyze Design
                          </>
                        )}
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {/* Score */}
                      <div className="bg-zinc-900 p-6 rounded-2xl shadow border border-zinc-800">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-white font-bold text-lg">Design Score</h4>
                          <div
                            className={cn(
                              "px-4 py-2 rounded-2xl text-sm font-bold shadow-sm",
                              uploadedFile.analysis.score >= 90
                                ? "bg-emerald-600 text-white"
                                : uploadedFile.analysis.score >= 75
                                  ? "bg-blue-600 text-white"
                                  : "bg-indigo-600 text-white",
                            )}
                          >
                            {uploadedFile.analysis.score >= 90
                              ? "Excellent"
                              : uploadedFile.analysis.score >= 75
                                ? "Good"
                                : "Needs Work"}
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="flex-1 bg-slate-600 rounded-full h-3 overflow-hidden">
                            <div
                              className={cn(
                                "h-3 rounded-full transition-all duration-1000 ease-out",
                                uploadedFile.analysis.score >= 90
                                  ? "bg-gradient-to-r from-emerald-500 to-emerald-400"
                                  : uploadedFile.analysis.score >= 75
                                    ? "bg-gradient-to-r from-blue-500 to-blue-400"
                                    : "bg-gradient-to-r from-indigo-500 to-pink-500",
                              )}
                              style={{ width: `${uploadedFile.analysis.score}%` }}
                            />
                          </div>
                          <span className="text-white font-bold text-xl">{uploadedFile.analysis.score}/100</span>
                        </div>
                      </div>

                      {/* Suggestions */}
                      <div className="bg-zinc-900 p-6 rounded-2xl shadow border border-zinc-800">
                        <h4 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
                          <div className="p-2 bg-blue-600 rounded-xl">
                            <Sparkles className="w-4 h-4 text-white" />
                          </div>
                          AI Suggestions
                        </h4>
                        <ul className="space-y-3">
                          {uploadedFile.analysis.suggestions.map((suggestion, i) => (
                            <li
                              key={i}
                              className="text-gray-300 flex items-start gap-3 p-4 bg-slate-600/50 rounded-2xl hover:bg-slate-600/70 transition-colors duration-200"
                            >
                              <span className="text-indigo-400 mt-1 font-bold">•</span>
                              <span className="leading-relaxed">{suggestion}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Improvements */}
                      <div className="bg-zinc-900 p-6 rounded-2xl shadow border border-zinc-800">
                        <h4 className="text-white font-bold text-lg mb-4 flex items-center gap-2">
                          <div className="p-2 bg-emerald-600 rounded-xl">
                            <TrendingUp className="w-4 h-4 text-white" />
                          </div>
                          Recommended Improvements
                        </h4>
                        <ul className="space-y-3">
                          {uploadedFile.analysis.improvements.map((improvement, i) => (
                            <li
                              key={i}
                              className="text-gray-300 flex items-start gap-3 p-4 bg-slate-600/50 rounded-2xl hover:bg-slate-600/70 transition-colors duration-200"
                            >
                              <span className="text-emerald-400 mt-1 font-bold">✓</span>
                              <span className="leading-relaxed">{improvement}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {uploadedFiles.length === 0 && (
        <Card className="bg-slate-800/80 backdrop-blur-sm p-12 text-center rounded-2xl shadow-lg">
          <div className="max-w-md mx-auto">
            <div className="p-6 bg-gradient-to-br from-slate-700 to-indigo-700 rounded-2xl w-fit mx-auto mb-6 shadow-lg">
              <ImageIcon className="w-12 h-12 text-indigo-300" />
            </div>
            <h3 className="text-xl font-bold text-white mb-3">No ads uploaded yet</h3>
            <p className="text-gray-300 leading-relaxed">
              Upload your first marketing advertisement to get started with AI-powered design analysis and optimization
              suggestions.
            </p>
          </div>
        </Card>
      )}
    </div>
  )
}
