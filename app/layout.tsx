import type { Metadata, Viewport } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'
import { AppProvider } from '@/components/app-context'
import { AuthProvider } from '@/components/auth-provider'
import { SiteHeader } from '@/components/site-header'
import { Suspense } from 'react'
import { Analytics } from '@vercel/analytics/react'

export const metadata: Metadata = {
  title: 'Madworks AI',
  description: 'Created with v0',
  generator: 'v0.app',
  icons: {
    icon: '/mwlg2.png',
    shortcut: '/mwlg2.png',
    apple: '/mwlg2.png',
  },
  manifest: '/site.webmanifest',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body className="text-zinc-100 min-h-screen overflow-x-hidden antialiased">
        <AuthProvider>
          <AppProvider>
            <div className="relative min-h-screen flex flex-col">
              {/* Global subtle background gradient (behind everything, including header) */}
              <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
                {/* Subtle marble texture overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-900/20 via-pink-900/10 to-orange-900/20 opacity-30"></div>
                {/* Flowing organic shapes */}
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-br from-indigo-600/5 to-pink-600/5 rounded-full blur-3xl"></div>
                <div className="absolute top-1/2 right-1/3 w-80 h-80 bg-gradient-to-br from-pink-600/5 to-orange-600/5 rounded-full blur-3xl"></div>
                <div className="absolute bottom-1/4 left-1/3 w-[500px] h-[500px] bg-gradient-to-br from-orange-600/8 to-indigo-600/8 rounded-full blur-3xl"></div>
                <div className="absolute bottom-1/6 right-1/4 w-[400px] h-[400px] bg-gradient-to-br from-indigo-600/8 to-pink-600/8 rounded-full blur-3xl"></div>
                {/* Veining patterns */}
                <div className="absolute inset-0 bg-gradient-to-br from-transparent via-indigo-500/2 to-transparent"></div>
                <div className="absolute inset-0 bg-gradient-to-tl from-transparent via-pink-500/2 to-transparent"></div>
              </div>

              <Suspense fallback={null}>
                <SiteHeader />
              </Suspense>
              <main className="flex-1 min-h-0" style={{ paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}>
                {children}
              </main>
              <Analytics />
            </div>
          </AppProvider>
        </AuthProvider>
      </body>
    </html>
  )
}
