"use client"

import React, { createContext, useContext, useMemo, useState } from "react"

export type OutputVariant = { format: string; w: number; h: number; url: string }

export type GalleryItem = {
  id: string
  title: string
  thumbnail_url?: string
  createdAt: number
  variants: OutputVariant[]
}

type AppContextShape = {
  credits: number
  setCredits: (n: number) => void
  decrementCredit: (by?: number) => void
  gallery: GalleryItem[]
  addToGallery: (item: GalleryItem) => void
  currentProject: string
  setCurrentProject: (p: string) => void
}

const AppContext = createContext<AppContextShape | null>(null)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [credits, setCredits] = useState<number>(100)
  const [gallery, setGallery] = useState<GalleryItem[]>([])
  const [currentProject, setCurrentProject] = useState<string>("Default Project")

  const decrementCredit = (by = 1) => setCredits((c) => Math.max(0, c - by))
  const addToGallery = (item: GalleryItem) => setGallery((g) => [item, ...g])

  const value = useMemo(
    () => ({ credits, setCredits, decrementCredit, gallery, addToGallery, currentProject, setCurrentProject }),
    [credits, gallery, currentProject]
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error("useApp must be used within AppProvider")
  return ctx
}


