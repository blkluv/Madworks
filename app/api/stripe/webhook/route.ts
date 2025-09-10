import { NextResponse } from "next/server";
import Stripe from "stripe";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Placeholder provisioning: replace with your real user upgrade logic
async function markUserUpgraded(userId: string, session: Stripe.Checkout.Session) {
  console.log(`[stripe:webhook] Marking user upgraded`, { userId, sessionId: session.id });
  // TODO: integrate with your user store to persist upgrade, e.g.,
  // await db.user.update({ where: { id: userId }, data: { isUpgraded: true } })
}

export async function POST(req: Request) {
  const sig = req.headers.get("stripe-signature");
  if (!process.env.STRIPE_WEBHOOK_SECRET) {
    console.error("Missing STRIPE_WEBHOOK_SECRET env var");
    return NextResponse.json({ error: "Webhook not configured" }, { status: 500 });
  }
  if (!process.env.STRIPE_SECRET_KEY) {
    console.error("Missing STRIPE_SECRET_KEY env var");
    return NextResponse.json({ error: "Stripe not configured" }, { status: 500 });
  }

  const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
    apiVersion: "2024-06-20",
  });

  let event: Stripe.Event;
  const buf = Buffer.from(await req.arrayBuffer());
  try {
    event = stripe.webhooks.constructEvent(buf, sig as string, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err: any) {
    console.error("Webhook signature verification failed.", err?.message);
    return NextResponse.json({ error: `Webhook Error: ${err.message}` }, { status: 400 });
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object as Stripe.Checkout.Session;
        const userId = (session.metadata?.userId as string) || "unknown";
        await markUserUpgraded(userId, session);
        break;
      }
      default: {
        // no-op for other events
        break;
      }
    }

    return NextResponse.json({ received: true });
  } catch (err: any) {
    console.error("Error handling webhook", err);
    return NextResponse.json({ error: err?.message || "unknown" }, { status: 500 });
  }
}
