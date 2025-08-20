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

// Reuse global in-memory store created in list route
const store: Map<string, Conversation[]> = (globalThis as any).__conv_store__ || new Map<string, Conversation[]>();
// @ts-ignore
if (!(globalThis as any).__conv_store__) (globalThis as any).__conv_store__ = store;

export async function GET(_: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = await auth();
  if (!session?.user?.email) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const email = session.user.email as string;
  const { id } = await ctx.params;
  const list = store.get(email) || [];
  const conv = list.find((c) => c.id === id);
  if (!conv) return NextResponse.json({ error: "Not found" }, { status: 404 });
  return NextResponse.json({ conversation: conv });
}

export async function PUT(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = await auth();
  if (!session?.user?.email) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const email = session.user.email as string;
  const body = await req.json().catch(() => ({}));
  const now = new Date().toISOString();
  const { id } = await ctx.params;
  const incoming: Conversation = {
    id: String(id),
    title: String(body?.title || "Untitled"),
    messages: Array.isArray(body?.messages) ? body.messages : [],
    createdAt: String(body?.createdAt || now),
    updatedAt: now,
    analysis: body?.analysis ?? undefined,
  };
  const list = store.get(email) || [];
  const filtered = list.filter((c) => c.id !== incoming.id);
  filtered.unshift(incoming);
  store.set(email, filtered);
  return NextResponse.json({ ok: true, conversation: incoming });
}

export async function DELETE(_: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = await auth();
  if (!session?.user?.email) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const email = session.user.email as string;
  const { id } = await ctx.params;
  const list = store.get(email) || [];
  const next = list.filter((c) => c.id !== id);
  store.set(email, next);
  return NextResponse.json({ ok: true });
}
