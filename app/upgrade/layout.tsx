"use client"

import { SiteHeader } from "@/components/site-header"
import { useApp } from "@/components/app-context"

export default function UpgradeLayout({ children }: { children: React.ReactNode }) {
  const { credits } = useApp()
  return (
    <>
      <SiteHeader credits={credits} />
      {children}
    </>
  )
}


