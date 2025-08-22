import type { NextAuthConfig } from "next-auth"
import Google from "next-auth/providers/google"

// Auth.js / NextAuth v5 configuration
const authConfig = {
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      authorization: {
        params: {
          redirect_uri: 'https://www.madworksai.com/api/auth/callback/google'
        }
      }
    }),
  ],
  session: { strategy: "jwt" },
  pages: {
    signIn: "/login",
  },
  trustHost: true,
} satisfies NextAuthConfig

export default authConfig
