import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { loadConversations, upsertConversation, type Conversation } from "./_store";

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Conversations are persisted per Google account via a simple file-based store.
// For production, use a durable DB (e.g., Postgres/SQLite/Firestore).

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
    const conversations = await loadConversations(email);
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
      id: String(body?.id || `c_${Date.now()}_${Math.random().toString(36).slice(2)}`),
      title: String(body?.title || "New chat"),
      messages: Array.isArray(body?.messages) ? body.messages : [],
      createdAt: String(body?.createdAt || now),
      updatedAt: now,
      analysis: body?.analysis ?? undefined,
    };

    const saved = await upsertConversation(email, conv);

    return new NextResponse(
      JSON.stringify({ conversation: saved }), 
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
