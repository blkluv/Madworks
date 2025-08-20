import { NextResponse } from "next/server";
import { auth } from "@/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  attachments?: Array<{ type: string; url: string }>;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
  analysis?: any;
}

// Simple in-memory store keyed by user email. NOTE: ephemeral, not production-grade.
const store: Map<string, Conversation[]> = (globalThis as any).__conv_store__ || new Map<string, Conversation[]>();
if (!(globalThis as any).__conv_store__) (globalThis as any).__conv_store__ = store;

export async function GET() {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const email = session.user.email as string;
  const conversations = store.get(email) || [];
  return NextResponse.json({ conversations });
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const email = session.user.email as string;
  const body = await req.json().catch(() => ({}));
  const now = new Date().toISOString();
  const conv: Conversation = {
    id: body?.id || `c_${Date.now()}`,
    title: body?.title || "New chat",
    messages: Array.isArray(body?.messages) ? body.messages : [],
    createdAt: body?.createdAt || now,
    updatedAt: now,
    analysis: body?.analysis ?? undefined,
  };
  const list = store.get(email) || [];
  store.set(email, [conv, ...list.filter((c) => c.id !== conv.id)]);
  return NextResponse.json({ ok: true, conversation: conv });
}
