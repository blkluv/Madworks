import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { getConversation, upsertConversation, deleteConversation, type Conversation } from "../_store";

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Use file-based store shared with list route. For production, replace with durable DB.

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
    const conv = await getConversation(email, id);
    
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

    const saved = await upsertConversation(email, incoming);

    return new NextResponse(
      JSON.stringify({ conversation: saved }), 
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
    await deleteConversation(email, id);
    
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
