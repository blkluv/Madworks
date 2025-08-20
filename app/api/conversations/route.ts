import { NextResponse } from "next/server";
import { auth } from "@/auth";

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
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

// Simple in-memory store keyed by user email. NOTE: ephemeral, not production-grade.
const store: Map<string, Conversation[]> = (globalThis as any).__conv_store__ || new Map<string, Conversation[]>();
if (!(globalThis as any).__conv_store__) (globalThis as any).__conv_store__ = store;

export async function OPTIONS() {
  return new NextResponse(null, { headers: CORS_HEADERS });
}

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.email) {
      return new NextResponse(
        JSON.stringify({ error: 'Unauthorized' }), 
        { status: 401, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
      );
    }
    const email = session.user.email as string;
    const conversations = store.get(email) || [];
    return new NextResponse(
      JSON.stringify({ conversations }), 
      { status: 200, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  } catch (error) {
    console.error('Error in GET /api/conversations:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal Server Error' }), 
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}

export async function POST(req: Request) {
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
    const conv: Conversation = {
      id: body?.id || `c_${Date.now()}`,
      title: body?.title || "New chat",
      messages: Array.isArray(body?.messages) ? body.messages : [],
      createdAt: body?.createdAt || now,
      updatedAt: now,
    };

    const conversations = store.get(email) || [];
    conversations.push(conv);
    store.set(email, conversations);

    return new NextResponse(
      JSON.stringify({ conversation: conv }), 
      { status: 201, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  } catch (error) {
    console.error('Error in POST /api/conversations:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Internal Server Error' }), 
      { status: 500, headers: { 'Content-Type': 'application/json', ...CORS_HEADERS } }
    );
  }
}
