"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Settings, Image, Monitor, Palette } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export default function SettingsPage() {
  return (
    <div className="min-h-screen text-zinc-100">
      <div className="container mx-auto px-4 py-10">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25">
              <Settings className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold">Settings</h1>
          </div>

          <Tabs defaultValue="appearance" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-8">
              <TabsTrigger value="appearance" className="flex items-center gap-2">
                <Palette className="w-4 h-4" />
                <span>Appearance</span>
              </TabsTrigger>
              <TabsTrigger value="output" className="flex items-center gap-2">
                <Image className="w-4 h-4" />
                <span>Output Settings</span>
              </TabsTrigger>
              <TabsTrigger value="account" className="flex items-center gap-2">
                <Monitor className="w-4 h-4" />
                <span>Account</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="appearance">
              <Card className="border-zinc-800 bg-zinc-950/80 backdrop-blur">
                <CardHeader>
                  <CardTitle>Appearance</CardTitle>
                  <CardDescription>Customize how Madworks looks on your device.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Dark Mode</Label>
                      <p className="text-sm text-zinc-400">Switch between light and dark theme</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Reduced Motion</Label>
                      <p className="text-sm text-zinc-400">Reduce animations and transitions</p>
                    </div>
                    <Switch />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="output">
              <Card className="border-zinc-800 bg-zinc-950/80 backdrop-blur">
                <CardHeader>
                  <CardTitle>Default Output Settings</CardTitle>
                  <CardDescription>Configure your default generation preferences</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label>Default Aspect Ratio</Label>
                    <Select defaultValue="3:4">
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select aspect ratio" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="3:4">3:4 (1080x1440) - Default</SelectItem>
                        <SelectItem value="1:1">1:1 (1080x1080) - Square</SelectItem>
                        <SelectItem value="9:16">9:16 (1080x1920) - Reels</SelectItem>
                        <SelectItem value="16:9">16:9 (1920x1080) - Landscape</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-sm text-zinc-400">Default aspect ratio for new generations</p>
                  </div>

                  <div className="space-y-2">
                    <Label>Output Quality</Label>
                    <Select defaultValue="high">
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select quality" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard (Faster)</SelectItem>
                        <SelectItem value="high">High Quality</SelectItem>
                        <SelectItem value="ultra">Ultra HD (Slower)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="account">
              <Card className="border-zinc-800 bg-zinc-950/80 backdrop-blur">
                <CardHeader>
                  <CardTitle>Account Settings</CardTitle>
                  <CardDescription>Manage your account preferences</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label>Email Notifications</Label>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Product updates</span>
                        <Switch defaultChecked />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Newsletter</span>
                        <Switch defaultChecked />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Account activity</span>
                        <Switch defaultChecked />
                      </div>
                    </div>
                  </div>

                  <div className="pt-4">
                    <Button variant="outline" className="w-full">
                      Sign out all devices
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
