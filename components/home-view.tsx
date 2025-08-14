"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FolderOpen, BookOpen, Users, Crown, Check, Sparkles, TrendingUp, Zap, Shield } from "lucide-react"

export function HomeView() {
  return (
    <div className="max-w-6xl mx-auto space-y-12">
      {/* Hero Section */}
      <Card className="bg-slate-800/90 backdrop-blur-sm overflow-hidden relative shadow-2xl rounded-3xl">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-800/80 via-indigo-900/40 to-indigo-900/60 opacity-90"></div>
        <div className="relative p-12 text-center">
          <div className="mx-auto mb-8 p-6 rounded-3xl bg-gradient-to-br from-slate-700 via-indigo-700 to-indigo-700 w-fit shadow-2xl">
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-indigo-400/20 opacity-60 rounded-3xl"></div>
            <Sparkles className="h-16 w-16 text-indigo-300 relative z-10" />
          </div>

          <h1 className="text-4xl font-bold text-white mb-6 tracking-tight">Transform Your Ads with AI</h1>
          <p className="text-gray-300 text-xl mb-8 leading-relaxed font-medium max-w-3xl mx-auto">
            Upload your marketing advertisements and get instant AI-powered design analysis, performance insights, and
            actionable optimization recommendations to boost your campaign success.
          </p>

          <div className="flex flex-wrap justify-center gap-4">
            <div className="flex items-center gap-2 bg-slate-700/80 px-4 py-2 rounded-2xl">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              <span className="text-white font-medium">Performance Analytics</span>
            </div>
            <div className="flex items-center gap-2 bg-slate-700/80 px-4 py-2 rounded-2xl">
              <Zap className="w-5 h-5 text-yellow-400" />
              <span className="text-white font-medium">Instant Analysis</span>
            </div>
            <div className="flex items-center gap-2 bg-slate-700/80 px-4 py-2 rounded-2xl">
              <Shield className="w-5 h-5 text-blue-400" />
              <span className="text-white font-medium">Secure & Private</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Dashboard Overview */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-2xl shadow-lg group-hover:scale-110 transition-transform duration-300">
              <Upload className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-white font-bold text-lg">Upload</h3>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            Drag and drop your ads to get instant AI analysis with performance insights and optimization suggestions.
          </p>
          <div className="flex items-center gap-2 text-indigo-400 text-sm font-medium">
            <Sparkles className="w-4 h-4" />
            <span>AI-Powered Analysis</span>
          </div>
        </Card>

        <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-blue-600 to-blue-700 rounded-2xl shadow-lg group-hover:scale-110 transition-transform duration-300">
              <FolderOpen className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-white font-bold text-lg">Projects</h3>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            Manage all your uploads, conversations, and AI-generated media in one organized workspace.
          </p>
          <div className="flex items-center gap-2 text-blue-400 text-sm font-medium">
            <TrendingUp className="w-4 h-4" />
            <span>Project Management</span>
          </div>
        </Card>

        <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-lg group-hover:scale-110 transition-transform duration-300">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-white font-bold text-lg">Gallery</h3>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            Save your favorite community templates and showcase your published projects in a beautiful gallery.
          </p>
          <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
            <BookOpen className="w-4 h-4" />
            <span>Personal Collection</span>
          </div>
        </Card>

        <Card className="bg-slate-800/90 backdrop-blur-sm p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-pink-600 to-rose-600 rounded-2xl shadow-lg group-hover:scale-110 transition-transform duration-300">
              <Users className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-white font-bold text-lg">Community</h3>
          </div>
          <p className="text-gray-300 text-sm leading-relaxed mb-4">
            Discover popular templates uploaded by other users and get inspired by trending designs.
          </p>
          <div className="flex items-center gap-2 text-pink-400 text-sm font-medium">
            <Users className="w-4 h-4" />
            <span>Community Templates</span>
          </div>
        </Card>
      </div>

      {/* Pricing Plans */}
      <div className="space-y-8">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-white mb-4">Choose Your Plan</h2>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Get started for free or unlock advanced features with our premium plans
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Free Plan */}
          <Card className="bg-slate-800/90 backdrop-blur-sm p-8 rounded-2xl shadow-lg relative">
            <div className="text-center mb-6">
              <h3 className="text-xl font-bold text-white mb-2">Free</h3>
              <div className="text-3xl font-bold text-white mb-1">$0</div>
              <p className="text-gray-400">per month</p>
            </div>

            <ul className="space-y-3 mb-8">
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>5 uploads per month</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Basic AI analysis</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Community templates</span>
              </li>
            </ul>

            <Button className="w-full bg-slate-700 hover:bg-slate-600 text-white rounded-2xl py-3 font-semibold">
              Get Started
            </Button>
          </Card>

          {/* Pro Plan */}
          <Card className="bg-gradient-to-br from-indigo-900/50 to-pink-900/50 backdrop-blur-sm p-8 rounded-2xl shadow-2xl relative border border-indigo-500/30">
            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
              <div className="bg-gradient-to-r from-indigo-600 to-pink-600 text-white px-4 py-1 rounded-full text-sm font-bold">
                Most Popular
              </div>
            </div>

            <div className="text-center mb-6">
              <h3 className="text-xl font-bold text-white mb-2">Pro</h3>
              <div className="text-3xl font-bold text-white mb-1">$19</div>
              <p className="text-gray-400">per month</p>
            </div>

            <ul className="space-y-3 mb-8">
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Unlimited uploads</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Advanced AI analysis</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>AI variations (up to 5)</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Priority support</span>
              </li>
            </ul>

            <Button className="w-full bg-gradient-to-r from-indigo-600 to-pink-600 hover:from-indigo-700 hover:to-pink-700 text-white rounded-2xl py-3 font-semibold shadow-lg">
              Upgrade to Pro
            </Button>
          </Card>

          {/* Enterprise Plan */}
          <Card className="bg-slate-800/90 backdrop-blur-sm p-8 rounded-2xl shadow-lg relative">
            <div className="text-center mb-6">
              <h3 className="text-xl font-bold text-white mb-2 flex items-center justify-center gap-2">
                <Crown className="w-5 h-5 text-yellow-400" />
                Enterprise
              </h3>
              <div className="text-3xl font-bold text-white mb-1">$99</div>
              <p className="text-gray-400">per month</p>
            </div>

            <ul className="space-y-3 mb-8">
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Everything in Pro</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Team collaboration</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Custom AI training</span>
              </li>
              <li className="flex items-center gap-3 text-gray-300">
                <Check className="w-5 h-5 text-emerald-400" />
                <span>Dedicated support</span>
              </li>
            </ul>

            <Button className="w-full bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700 text-white rounded-2xl py-3 font-semibold shadow-lg">
              Contact Sales
            </Button>
          </Card>
        </div>
      </div>
    </div>
  )
}
