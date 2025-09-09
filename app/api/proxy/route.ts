import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

// Very small image proxy to avoid mixed-content/CORS issues in the browser.
// Usage: /api/proxy?u=<absolute-url>
// - Returns upstream bytes with the same content-type.
// - No caching by default to ensure instant refresh during dev.
// - Optional host restriction via PIPELINE_URL_ALLOW_HOST (comma/space-separated) to reduce SSRF risk.

const allowRaw = [
  process.env.PIPELINE_URL_ALLOW_HOST,
  process.env.NEXT_PUBLIC_PIPELINE_URL,
  process.env.NEXT_PUBLIC_PIPELINE_BASE,
  process.env.PUBLIC_BASE_URL, // for parity with python service
  "http://localhost:8010",
]
  .filter(Boolean)
  .join(" ");

const ALLOW_HOSTS = allowRaw
  .split(/[\s,]+/)
  .map((s) => s.trim())
  .filter(Boolean);

function isLoopbackHost(host: string) {
  const h = (host || "").toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "::1" || h === "0.0.0.0";
}

function portsMatch(allowPort: string, targetPort: string) {
  // If allowPort is empty, treat 80/443 as defaults
  if (!allowPort) {
    return targetPort === "" || targetPort === "80" || targetPort === "443";
  }
  return allowPort === targetPort;
}

function allowed(url: URL) {
  if (ALLOW_HOSTS.length === 0) return true;
  return ALLOW_HOSTS.some((base) => {
    try {
      const b = new URL(base);
      const sameHost = b.hostname === url.hostname;
      const bothLoopback = isLoopbackHost(b.hostname) && isLoopbackHost(url.hostname);
      const hostOk = sameHost || bothLoopback;
      return hostOk && portsMatch(b.port, url.port);
    } catch {
      return false;
    }
  });
}

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const raw = searchParams.get("u") || "";
    if (!raw) return NextResponse.json({ error: "missing param u" }, { status: 400 });

    let target: URL;
    try {
      target = new URL(raw);
    } catch {
      return NextResponse.json({ error: "u must be absolute URL" }, { status: 400 });
    }

    if (!allowed(target)) {
      return NextResponse.json({ error: "host not allowed" }, { status: 403 });
    }

    const upstream = await fetch(target.toString(), { cache: "no-store" });
    const ct = upstream.headers.get("content-type") || "application/octet-stream";
    const sc = upstream.status;
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      return new NextResponse(text, { status: sc, headers: { "content-type": ct } });
    }

    const body = upstream.body;
    if (!body) {
      return new NextResponse(null, { status: sc, headers: { "content-type": ct } });
    }

    const res = new NextResponse(body, {
      status: sc,
      headers: {
        "content-type": ct,
        "cache-control": "no-store, max-age=0",
        "access-control-allow-origin": "*",
      },
    });
    return res;
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || "proxy error" }, { status: 500 });
  }
}
