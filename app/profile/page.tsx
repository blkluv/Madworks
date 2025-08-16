import { auth } from "@/auth"
import { redirect } from "next/navigation"
import { Button } from "@/components/ui/button"

export default async function ProfilePage() {
  const session = await auth()
  if (!session) {
    redirect("/login?callbackUrl=/profile")
  }

  const user = session.user

  return (
    <div className="min-h-screen text-zinc-100">
      <div className="container mx-auto px-4 py-10">
        <div className="max-w-2xl mx-auto bg-zinc-950/80 border border-zinc-900 rounded-3xl p-8 backdrop-blur">
          <div className="flex items-center gap-4 mb-6">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {user?.image ? (
              <img src={user.image} alt={user?.name ?? "User"} className="w-16 h-16 rounded-full object-cover" />
            ) : (
              <div className="w-16 h-16 rounded-full bg-zinc-800" />
            )}
            <div>
              <h1 className="text-2xl font-bold">{user?.name ?? "User"}</h1>
              <p className="text-zinc-400">{user?.email}</p>
            </div>
          </div>

          <div className="space-y-2 text-zinc-300">
            <p>Your account is connected with Google.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
