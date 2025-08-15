import { NextResponse } from "next/server";

// Endpoint orchestrates the Python pipeline service with extensive step-by-step logging.
// Accepts multipart/form-data (preferred) with fields:
// - prompt (string)
// - tone (optional string)
// - platform (optional string)
// - num_variants (optional number)
// - temperature (optional number)
// - crop_width, crop_height (optional numbers)
// - image (optional File)
// Also accepts JSON body with the same fields (no image in that case).

export const runtime = "edge";

const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || process.env.PIPELINE_URL || "http://localhost:8010";

function nowIso() {
  return new Date().toISOString();
}

type LogEntry = {
  step: string;
  status: "start" | "ok" | "error";
  message?: string;
  error?: string;
  timestamp: string;
  durationMs?: number;
  meta?: any;
};

function logStep(logs: LogEntry[], step: string, status: LogEntry["status"], message?: string, meta?: any, durationMs?: number, error?: any) {
  const entry: LogEntry = {
    step,
    status,
    message,
    timestamp: nowIso(),
    durationMs,
    meta,
  };
  if (error) entry.error = typeof error === "string" ? error : (error?.message || JSON.stringify(error));
  // eslint-disable-next-line no-console
  console.log(`[pipeline] ${step} - ${status}${message ? `: ${message}` : ""}`, meta || "");
  logs.push(entry);
}

async function fetchJson(url: string, init?: RequestInit) {
  const res = await fetch(url, init);
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} for ${url} -> ${txt}`);
  }
  return res.json();
}

export async function POST(req: Request) {
  const logs: LogEntry[] = [];
  const started = Date.now();
  let analysis: any = null;
  let copy: any = null;
  let composition: any = null;
  let renderRes: any = null;
  let qaRes: any = null;
  let exportRes: any = null;
  let ok = true;

  let prompt = "";
  let tone: string | undefined;
  let platform: string | undefined;
  let num_variants = 1;
  let temperature = 0.7;
  let crop_width = 1080;
  let crop_height = 1080;
  let imageFile: File | undefined;

  try {
    // Try to parse as multipart first
    let form: FormData | null = null;
    try {
      form = await req.formData();
    } catch (_) {
      form = null;
    }
    if (form) {
      prompt = String(form.get("prompt") || "");
      tone = form.get("tone") ? String(form.get("tone")) : undefined;
      platform = form.get("platform") ? String(form.get("platform")) : undefined;
      num_variants = form.get("num_variants") ? Number(form.get("num_variants")) : 1;
      temperature = form.get("temperature") ? Number(form.get("temperature")) : 0.7;
      crop_width = form.get("crop_width") ? Number(form.get("crop_width")) : 1080;
      crop_height = form.get("crop_height") ? Number(form.get("crop_height")) : 1080;
      const maybeFile = form.get("image");
      if (maybeFile && maybeFile instanceof File) {
        imageFile = maybeFile;
      }
    } else {
      const json = await req.json().catch(() => ({}));
      prompt = String(json.prompt || "");
      tone = json.tone;
      platform = json.platform;
      num_variants = json.num_variants ?? 1;
      temperature = json.temperature ?? 0.7;
      crop_width = json.crop_width ?? 1080;
      crop_height = json.crop_height ?? 1080;
      // No image via JSON
    }
  } catch (e) {
    ok = false;
    logStep(logs, "parse_input", "error", "Failed to parse input", undefined, undefined, e);
    return NextResponse.json({ ok, logs, error: "Bad input" }, { status: 400 });
  }

  const job_id = `web_${Math.random().toString(36).slice(2, 10)}`;

  // Step 1: Ingest + Analyze (optional if image present)
  if (imageFile) {
    const t0 = Date.now();
    logStep(logs, "ingest_analyze", "start", "Uploading image to pipeline");
    try {
      const f = new FormData();
      f.append("image", imageFile, imageFile.name || "upload.png");
      analysis = await fetchJson(`${PIPELINE_URL}/ingest-analyze`, { method: "POST", body: f });
      logStep(logs, "ingest_analyze", "ok", "Image analyzed", { mask_url: analysis?.mask_url }, Date.now() - t0);
    } catch (e) {
      ok = false; // non-fatal, can proceed without analysis
      logStep(logs, "ingest_analyze", "error", "Failed to analyze image", undefined, Date.now() - t0, e);
    }
  } else {
    logStep(logs, "ingest_analyze", "ok", "No image provided, skipping analysis");
  }

  // Step 2: Copy generation
  const t1 = Date.now();
  logStep(logs, "copy", "start", "Generating ad copy variants");
  try {
    const copyPayload = {
      copy_instructions: prompt,
      facts: {},
      constraints: undefined,
      num_variants,
      temperature,
      tone,
      platform,
    };
    copy = await fetchJson(`${PIPELINE_URL}/copy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(copyPayload),
    });
    logStep(logs, "copy", "ok", "Copy generated", { variant_count: copy?.variants?.length || 1 }, Date.now() - t1);
  } catch (e) {
    ok = false;
    logStep(logs, "copy", "error", "Failed to generate copy", undefined, Date.now() - t1, e);
  }

  // Step 3: Compose SVG
  const t2 = Date.now();
  logStep(logs, "compose", "start", "Composing SVG");
  try {
    const crop_info = { width: crop_width, height: crop_height };
    const useCopy = copy?.variants?.[0] || copy || { headline: "", subheadline: "", cta: "Learn More" };
    const composePayload = {
      copy: useCopy,
      analysis: analysis || {},
      crop_info,
    };
    composition = await fetchJson(`${PIPELINE_URL}/compose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(composePayload),
    });
    logStep(logs, "compose", "ok", "SVG composed", { composition_id: composition?.composition_id }, Date.now() - t2);
  } catch (e) {
    ok = false;
    logStep(logs, "compose", "error", "Failed to compose SVG", undefined, Date.now() - t2, e);
  }

  // Step 4: Render
  const t3 = Date.now();
  logStep(logs, "render", "start", "Rendering outputs");
  try {
    const renderPayload = {
      composition,
      crop_info: { width: crop_width, height: crop_height },
      job_id,
    };
    renderRes = await fetchJson(`${PIPELINE_URL}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(renderPayload),
    });
    logStep(logs, "render", "ok", "Rendered outputs", { outputs: renderRes?.outputs?.length }, Date.now() - t3);
  } catch (e) {
    ok = false;
    logStep(logs, "render", "error", "Failed to render outputs", undefined, Date.now() - t3, e);
  }

  // Step 5: QA
  const t4 = Date.now();
  logStep(logs, "qa", "start", "Running QA checks");
  try {
    const qaPayload = {
      composition,
      copy,
      render: renderRes,
    };
    qaRes = await fetchJson(`${PIPELINE_URL}/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(qaPayload),
    });
    logStep(logs, "qa", "ok", "QA completed", { ok: qaRes?.ok }, Date.now() - t4);
  } catch (e) {
    ok = false;
    logStep(logs, "qa", "error", "QA failed", undefined, Date.now() - t4, e);
  }

  // Step 6: Export
  const t5 = Date.now();
  logStep(logs, "export", "start", "Exporting manifest");
  try {
    const exportPayload = {
      render: renderRes,
      job_id,
      timestamp: nowIso(),
    };
    exportRes = await fetchJson(`${PIPELINE_URL}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportPayload),
    });
    logStep(logs, "export", "ok", "Export completed", { manifest_url: exportRes?.manifest_url }, Date.now() - t5);
  } catch (e) {
    ok = false;
    logStep(logs, "export", "error", "Export failed", undefined, Date.now() - t5, e);
  }

  const totalMs = Date.now() - started;
  logStep(logs, "summary", ok ? "ok" : "error", `Pipeline finished in ${totalMs}ms`);

  return NextResponse.json({
    ok,
    job_id,
    copy_best: copy?.headline ? { headline: copy.headline, subheadline: copy.subheadline, cta: copy.cta } : (copy?.variants?.[0] || null),
    copy_variants: copy?.variants || (copy ? [copy] : []),
    composition,
    render: renderRes,
    qa: qaRes,
    export: exportRes,
    logs,
    thumbnail_url: renderRes?.thumbnail_url,
    outputs: renderRes?.outputs || [],
  });
}
