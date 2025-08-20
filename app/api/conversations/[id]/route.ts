import { NextResponse } from "next/server";
import { auth } from "@/auth";

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

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

export async function OPTIONS() {
  return new NextResponse(null, { headers: CORS_HEADERS });
}

export async function GET(_: Request, ctx: { params: Promise<{ id: string }> }) {
  try {
    const session = await auth();
    if (!session?.user?.email) {
      return new NextResponse(
        JSON.stringify({ error: 'Unauthorized' }), 
        { status: 401, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }
    const email = session.user.email as string;
    const { id } = await ctx.params;
    const list = store.get(email) || [];
    const conv = list.find((c) => c.id === id);
    
    if (!conv) {
      return new NextResponse(
        JSON.stringify({ error: 'Not found' }), 
        { status: 404, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }
    
    return new NextResponse(
      JSON.stringify({ conversation: conv }), 
      { status: 200, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  } catch (error) {
    console.error('Error in GET /api/conversations/[id]:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal Server Error' }), 
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}

export async function PUT(req: Request, ctx: { params: Promise<{ id: string }> }) {
  try {
    const session = await auth();
    if (!session?.user?.email) {
      return new NextResponse(
        JSON.stringify({ error: 'Unauthorized' }), 
        { status: 401, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }
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
    const index = list.findIndex((c) => c.id === id);
    
    if (index === -1) {
      return new NextResponse(
        JSON.stringify({ error: 'Not found' }), 
        { status: 404, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }

    list[index] = incoming;
    store.set(email, list);

    return new NextResponse(
      JSON.stringify({ conversation: incoming }), 
      { status: 200, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  } catch (error) {
    console.error('Error in PUT /api/conversations/[id]:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal Server Error' }), 
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}

export async function DELETE(_: Request, ctx: { params: Promise<{ id: string }> }) {
  try {
    const session = await auth();
    if (!session?.user?.email) {
      return new NextResponse(
        JSON.stringify({ error: 'Unauthorized' }), 
        { status: 401, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }
    
    const email = session.user.email as string;
    const { id } = await ctx.params;
    const list = store.get(email) || [];
    const filtered = list.filter((c) => c.id !== id);
    store.set(email, filtered);
    
    return new NextResponse(null, { 
      status: 204, 
      headers: { ...CORS_HEADERS } 
    });
  } catch (error) {
    console.error('Error in DELETE /api/conversations/[id]:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal Server Error' }), 
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}
