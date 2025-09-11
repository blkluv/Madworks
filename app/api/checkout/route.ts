import { NextResponse } from "next/server";
import Stripe from "stripe";
import { auth } from "@/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function getSiteUrl() {
  const siteUrl = process.env.SITE_URL || process.env.NEXT_PUBLIC_SITE_URL;
  return siteUrl?.replace(/\/$/, "");
}

async function getOrCreateCustomer(stripe: Stripe, email: string) {
  // Try to find existing customer by email
  const existing = await stripe.customers.list({ email, limit: 1 });
  if (existing.data.length > 0) return existing.data[0];
  // Create a new customer to carry metadata like free_used/is_upgraded
  return stripe.customers.create({ email, metadata: {} });
}

export async function POST(req: Request) {
  try {
    const session = await auth();
    if (!process.env.STRIPE_SECRET_KEY) {
      return NextResponse.json({ error: "Stripe not configured" }, { status: 500 });
    }
    const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
      apiVersion: "2024-06-20",
    });

    // Expect a price_id from env (for Starter Kit)
    const priceId = process.env.STRIPE_STARTER_PRICE_ID;
    if (!priceId) {
      return NextResponse.json({ error: "Missing STRIPE_STARTER_PRICE_ID" }, { status: 500 });
    }

    const body = await req.json().catch(() => ({}));
    const quantity = typeof body?.quantity === "number" && body.quantity > 0 ? body.quantity : 1;

    const siteUrl = getSiteUrl();
    if (!siteUrl) {
      return NextResponse.json({ error: "Missing SITE_URL or NEXT_PUBLIC_SITE_URL" }, { status: 500 });
    }

    const userId = (session?.user as any)?.id || (session?.user as any)?.email || "anonymous";
    const customerEmail = (session?.user as any)?.email;

    let customerId: string | undefined = undefined;
    if (customerEmail) {
      try {
        const customer = await getOrCreateCustomer(stripe, customerEmail);
        customerId = customer.id;
      } catch (e) {
        // Non-fatal, proceed without explicit customer linkage
      }
    }

    const baseParams: Stripe.Checkout.SessionCreateParams = {
      mode: "payment",
      line_items: [
        {
          price: priceId,
          quantity,
        },
      ],
      allow_promotion_codes: true,
      success_url: `${siteUrl}/upgrade/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${siteUrl}/upgrade/cancel`,
      metadata: {
        userId: String(userId),
      },
    };

    const checkoutSession = await stripe.checkout.sessions.create({
      ...baseParams,
      ...(customerId ? { customer: customerId } : { customer_email: customerEmail ?? undefined }),
    });

    if (!checkoutSession.url) {
      return NextResponse.json({ error: "Failed to create checkout session" }, { status: 500 });
    }

    return NextResponse.json({ url: checkoutSession.url });
  } catch (err: any) {
    console.error("/api/checkout error", err);
    return NextResponse.json({ error: err?.message || "Unknown error" }, { status: 500 });
  }
}
