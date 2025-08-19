import { Suspense } from 'react'
import HomeContent from './home-content'

export default function HomePage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse">Loading...</div>
      </div>
    }>
      <HomeContent />
    </Suspense>
  )
}

