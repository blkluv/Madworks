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

const TIMEOUTS = {
  ingestAnalyze: 45_000,
  copy: 90_000,
  compose: 30_000,
  render: 120_000,
  qa: 30_000,
  export: 30_000,
};

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

async function fetchJson(url: string, init?: RequestInit, timeoutMs: number = 60_000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...(init || {}), signal: controller.signal });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText} for ${url} -> ${txt}`);
    }
    return res.json();
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(`Timeout after ${timeoutMs}ms for ${url}`);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
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
  let sizes: Array<{ name?: string; width: number; height: number }> | null = null;
  let history: Array<{ role: string; content: string }> | null = null;

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
      const sizesRaw = form.get("sizes");
      if (sizesRaw && typeof sizesRaw === "string") {
        try {
          const parsed = JSON.parse(sizesRaw);
          if (Array.isArray(parsed)) {
            sizes = parsed
              .map((s: any) => ({ name: s?.name, width: Number(s?.width), height: Number(s?.height) }))
              .filter((s: any) => Number.isFinite(s.width) && Number.isFinite(s.height));
          }
        } catch (_) {}
      }
      const historyRaw = form.get("history");
      if (historyRaw && typeof historyRaw === "string") {
        try {
          const parsedH = JSON.parse(historyRaw);
          if (Array.isArray(parsedH)) {
            history = parsedH
              .map((h: any) => ({ role: String(h?.role || ""), content: String(h?.content || "") }))
              .filter((h: any) => h.role && typeof h.content === "string");
          }
        } catch (_) {}
      }
      const analysisRaw = form.get("analysis");
      if (analysisRaw && typeof analysisRaw === "string") {
        try {
          const parsedA = JSON.parse(analysisRaw);
          if (parsedA && typeof parsedA === "object") analysis = parsedA;
        } catch (_) {}
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
      if (Array.isArray(json.sizes)) {
        sizes = json.sizes
          .map((s: any) => ({ name: s?.name, width: Number(s?.width), height: Number(s?.height) }))
          .filter((s: any) => Number.isFinite(s.width) && Number.isFinite(s.height));
      }
      if (Array.isArray(json.history)) {
        history = json.history
          .map((h: any) => ({ role: String(h?.role || ""), content: String(h?.content || "") }))
          .filter((h: any) => h.role && typeof h.content === "string");
      }
      // No image via JSON
      if (json.analysis) {
        try {
          analysis = typeof json.analysis === "string" ? JSON.parse(json.analysis) : json.analysis;
        } catch (_) {
          analysis = null;
        }
      }
    }
  } catch (e) {
    ok = false;
    logStep(logs, "parse_input", "error", "Failed to parse input", undefined, undefined, e);
    return NextResponse.json({ ok, logs, error: "Bad input" }, { status: 400 });
  }

  const job_id = `web_${Math.random().toString(36).slice(2, 10)}`;

  // Step 1: Ingest + Analyze (use provided analysis when available and no new image)
  if (imageFile) {
    const t0 = Date.now();
    logStep(logs, "ingest_analyze", "start", "Uploading image to pipeline");
    try {
      const f = new FormData();
      f.append("image", imageFile, imageFile.name || "upload.png");
      analysis = await fetchJson(`${PIPELINE_URL}/ingest-analyze`, { method: "POST", body: f }, TIMEOUTS.ingestAnalyze);
      logStep(logs, "ingest_analyze", "ok", "Image analyzed", { mask_url: analysis?.mask_url }, Date.now() - t0);
    } catch (e) {
      ok = false; // non-fatal, can proceed without analysis
      logStep(logs, "ingest_analyze", "error", "Failed to analyze image", undefined, Date.now() - t0, e);
    }
  } else if (analysis) {
    logStep(logs, "ingest_analyze", "ok", "Using provided analysis", { provided: true });
  } else {
    logStep(logs, "ingest_analyze", "ok", "No image or analysis provided, using fallback background");
  }

  // Step 2: Copy generation
  const t1 = Date.now();
  logStep(logs, "copy", "start", "Generating ad copy variants");
  try {
    const copyPayload = {
      copy_instructions: prompt,
      facts: history ? { history } : {},
      constraints: undefined,
      num_variants,
      temperature,
      tone,
      platform,
    };
    copy = await fetchJson(
      `${PIPELINE_URL}/copy`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(copyPayload),
      },
      TIMEOUTS.copy,
    );
    logStep(logs, "copy", "ok", "Copy generated", { variant_count: copy?.variants?.length || 1 }, Date.now() - t1);
  } catch (e) {
    ok = false;
    logStep(logs, "copy", "error", "Failed to generate copy", undefined, Date.now() - t1, e);
  }

  // Step 3 + 4: Compose and Render for multiple aspect ratios
  const defaultSizes = [
    { name: "square", width: 1080, height: 1080 },
    { name: "portrait", width: 1080, height: 1350 },
    { name: "landscape", width: 1920, height: 1080 },
    { name: "story", width: 1080, height: 1920 },
  ];
  const sizesToUse = (sizes && sizes.length ? sizes : defaultSizes) as Array<{ name?: string; width: number; height: number }>;
  const combinedOutputs: any[] = []
  let combinedThumb: string | undefined = undefined
  try {
    const copyVariants = (Array.isArray(copy?.variants) && copy.variants.length
      ? copy.variants
      : [copy?.headline
          ? { headline: copy.headline, subheadline: copy.subheadline, cta: copy.cta, emphasis_ranges: copy?.emphasis_ranges, font_recommendations: copy?.font_recommendations }
          : { headline: "", subheadline: "", cta: "Learn More" }]);

    for (let vi = 0; vi < copyVariants.length; vi++) {
      const v = copyVariants[vi];
      const vLabel = `v${vi + 1}`;
      for (const s of sizesToUse) {
        const t2 = Date.now();
        logStep(logs, `compose_${s.name}_${vLabel}`, "start", `Composing SVG ${s.width}x${s.height} for ${vLabel}`);
        const composePayload = {
          copy: v,
          analysis: analysis || {},
          crop_info: { width: s.width, height: s.height },
        };
        const comp = await fetchJson(
          `${PIPELINE_URL}/compose`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(composePayload),
          },
          TIMEOUTS.compose,
        );
        if (!composition) composition = comp; // capture first for backward compat
        logStep(logs, `compose_${s.name}_${vLabel}`, "ok", "SVG composed", { composition_id: comp?.composition_id }, Date.now() - t2);

        const t3 = Date.now();
        logStep(logs, `render_${s.name}_${vLabel}`, "start", `Rendering ${s.width}x${s.height} for ${vLabel}`);
        const renderPayload = { composition: comp, crop_info: { width: s.width, height: s.height }, job_id };
        const r = await fetchJson(
          `${PIPELINE_URL}/render`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(renderPayload),
          },
          TIMEOUTS.render,
        );
        // Tag outputs with copy variant and size for client-side use
        const outs = (r?.outputs || []).map((o: any) => ({ ...o, variant: vLabel, size: s.name }));
        combinedOutputs.push(...outs);
        if (!combinedThumb) combinedThumb = r?.thumbnail_url;
        logStep(logs, `render_${s.name}_${vLabel}`, "ok", "Rendered outputs", { outputs: outs.length }, Date.now() - t3);
      }
    }

    renderRes = { outputs: combinedOutputs, thumbnail_url: combinedThumb };
  } catch (e) {
    ok = false;
    logStep(logs, "render_all", "error", "Failed composing/rendering variants", undefined, undefined, e);
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
    qaRes = await fetchJson(
      `${PIPELINE_URL}/qa`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(qaPayload),
      },
      TIMEOUTS.qa,
    );
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
    exportRes = await fetchJson(
      `${PIPELINE_URL}/export`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(exportPayload),
      },
      TIMEOUTS.export,
    );
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
    analysis,
  });
}
