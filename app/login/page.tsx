import { Button } from "@/components/ui/button"
import { signIn } from "@/auth"
import { Crown } from "lucide-react"

export default function LoginPage() {
  async function loginWithGoogle() {
    "use server"
    await signIn("google")
  }

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-zinc-950/80 border border-zinc-900 rounded-3xl p-8 backdrop-blur">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-gradient-to-br from-indigo-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-indigo-600/25">
            <div className="w-6 h-6 flex items-center justify-center">
              <span className="text-white font-bold text-xl">M</span>
            </div>
          </div>
          <h1 className="text-2xl font-bold">Sign in to Madworks</h1>
        </div>

        <form action={loginWithGoogle}>
          <Button type="submit" className="w-full h-12 rounded-xl bg-white text-black hover:bg-zinc-200">
            {/* Simple G glyph */}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" className="w-5 h-5 mr-2">
              <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12   s5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C33.64,6.053,29.055,4,24,4C12.955,4,4,12.955,4,24s8.955,20,20,20   s20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
              <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,16.108,19.007,13,24,13c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657   C33.64,6.053,29.055,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
              <path fill="#4CAF50" d="M24,44c4.995,0,9.58-2.053,12.89-5.351l-5.972-5.058C29.861,35.477,27.045,36.5,24,36.5   c-5.202,0-9.619-3.317-11.278-7.946l-6.536,5.036C9.505,39.556,16.227,44,24,44z"/>
              <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.59   c0.001-0.001,0.002-0.001,0.003-0.002l5.972,5.058C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
            </svg>
            Continue with Google
          </Button>
        </form>
      </div>
    </div>
  )
}
