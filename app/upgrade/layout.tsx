"use client"

export default function UpgradeLayout({ children }: { children: React.ReactNode }) {
  // Header is rendered globally in app/layout.tsx
  return <>{children}</>
}


