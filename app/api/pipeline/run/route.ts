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

export const runtime = "nodejs";
export const maxDuration = 300; // allow up to 5 minutes when deployed to platforms that honor this setting

const PIPELINE_URL = process.env.NEXT_PUBLIC_PIPELINE_URL || process.env.PIPELINE_URL || "http://localhost:8010";

const TIMEOUTS = {
  ingestAnalyze: 45_000,
  copy: 90_000,
  compose: 60_000,
  render: 120_000,
  qa: 30_000,
  export: 30_000,
};

function nowIso() {
  return new Date().toISOString();
}

// Deterministic 32-bit hash for seeds (Edge-safe, no Node crypto required)
function hash32(str: string): number { //
  let h = 2166136261 >>> 0; // FNV-1a basis
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  // Mix a bit
  h += (h << 13) >>> 0; h ^= h >>> 7; h += (h << 3) >>> 0; h ^= h >>> 17; h += (h << 5) >>> 0;
  return h >>> 0;
}

// Simple LCG RNG for reproducible variant choices
function makeRng(seed: number) {
  let s = seed >>> 0;
  return function rand() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return (s >>> 0) / 0x100000000;
  };
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
  let text_color_override: string | undefined;
  let panel_side: string | undefined;
  let variant_override: any = undefined;
  let variant_seed_override: number | undefined;

  try {
    // Decide parsing mode by Content-Type
    const contentType = (req.headers.get("content-type") || "").toLowerCase();
    const isMultipart = contentType.includes("multipart/form-data");
    let form: FormData | null = null;
    if (isMultipart) {
      try {
        form = await req.formData();
      } catch (_) {
        form = null;
      }
    }
    if (isMultipart && form) {
      prompt = String(form.get("prompt") || "");
      tone = form.get("tone") ? String(form.get("tone")) : undefined;
      platform = form.get("platform") ? String(form.get("platform")) : undefined;
      num_variants = 1; // Force single variant to generate exactly one output
      temperature = form.get("temperature") ? Number(form.get("temperature")) : 0.7;
      crop_width = form.get("crop_width") ? Number(form.get("crop_width")) : 1080;
      crop_height = form.get("crop_height") ? Number(form.get("crop_height")) : 1080;
      const maybeFile = form.get("image");
      if (maybeFile && maybeFile instanceof File) {
        imageFile = maybeFile;
      }
      const tco = form.get("text_color_override");
      if (typeof tco === "string" && tco.trim()) {
        text_color_override = tco.trim();
      }
      const ps = form.get("panel_side");
      if (typeof ps === "string" && (ps === "left" || ps === "right")) {
        panel_side = ps;
      }
      const vRaw = form.get("variant");
      if (typeof vRaw === "string" && vRaw.trim()) {
        try { variant_override = JSON.parse(vRaw); } catch {}
      }
      const vs = form.get("variant_seed");
      if (typeof vs === "string" && vs.trim() && !Number.isNaN(Number(vs))) {
        variant_seed_override = Number(vs);
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
      num_variants = 1; // Force single variant to generate exactly one output
      temperature = json.temperature ?? 0.7;
      crop_width = json.crop_width ?? 1080;
      crop_height = json.crop_height ?? 1080;
      if (typeof json.text_color_override === "string" && json.text_color_override.trim()) {
        text_color_override = json.text_color_override.trim();
      }
      if (typeof json.panel_side === "string" && (json.panel_side === "left" || json.panel_side === "right" || json.panel_side === "center")) {
        panel_side = json.panel_side;
      }
      if (json.variant && typeof json.variant === "object") {
        variant_override = json.variant;
      }
      if (typeof json.variant_seed === "number") {
        variant_seed_override = json.variant_seed;
      }
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

  // Step 3 + 4: Compose and Render for a single aspect ratio
  // Prefer the original uploaded image dimensions when available to match input format exactly
  let sizesToUse: Array<{ name?: string; width: number; height: number }> = []
  try {
    const osz = Array.isArray((analysis as any)?.original_size) ? (analysis as any).original_size : null
    const ow = Number(osz?.[0])
    const oh = Number(osz?.[1])
    if (Number.isFinite(ow) && Number.isFinite(oh) && ow > 0 && oh > 0) {
      sizesToUse = [{ name: "original", width: ow, height: oh }]
    }
  } catch {}
  if (!sizesToUse.length) {
    sizesToUse = [{ name: "square", width: 1080, height: 1080 }]
  }
  const combinedOutputs: any[] = []
  let combinedThumb: string | undefined = undefined
  try {
    // Use the first size (original when available) and generate EXACTLY THREE fixed presets
    const s = sizesToUse[0]
    const seedStr = `${job_id}|${s.name}|${s.width}x${s.height}`
    const seed = (variant_seed_override ?? hash32(seedStr))
    const v0 = (Array.isArray(copy?.variants) && copy.variants.length)
      ? copy.variants[0]
      : (copy?.headline ? { headline: copy.headline, subheadline: copy.subheadline, cta: copy.cta, emphasis_ranges: copy?.emphasis_ranges, font_recommendations: copy?.font_recommendations } : { headline: "", subheadline: "", cta: "Learn More" })

    // Three consistent, user-friendly presets (no scrims/cards)
    const presets: Array<{ id: string; name: string; variant: any }> = [
      {
        id: "center_bold",
        name: "Centered · Bold",
        variant: {
          panel_side: "center",
          text_align: "center",
          vertical_align: "middle",
          show_scrim: false,
          panel_style: "none",
          headline_fill: "solid",
          type_scale: 1.1,
          panel_width_factor: 0.92,
          cta_style: "pill",
          cta_width_mode: "auto",
          bg_fit: "slice",
          emotional_mode: "neutral",
        },
      },
      {
        id: "left_bold_vignette_bl",
        name: "Left · Bold · BL Vignette",
        variant: {
          panel_side: "left",
          text_align: "left",
          vertical_align: "middle",
          show_scrim: false,
          panel_style: "none",
          headline_fill: "solid",
          type_scale: 1.08,
          panel_width_factor: 0.52,
          cta_style: "pill",
          cta_width_mode: "auto",
          bg_fit: "slice",
          emotional_mode: "strong",
          // Enable bold bottom-left vignette shadow
          corner_shadow_bl: true,
          corner_shadow_strength: 1.25,
        },
      },
      {
        id: "top_bold_white",
        name: "Top · Bold White",
        variant: {
          panel_side: "center",
          text_align: "center",
          vertical_align: "top",
          show_scrim: false,
          panel_style: "none",
          headline_fill: "solid",
          type_scale: 1.02,
          panel_width_factor: 0.90,
          cta_style: "pill",
          cta_width_mode: "auto",
          bg_fit: "slice",
          emotional_mode: "neutral",
        },
      },
    ]

    const comps: Array<any> = []
    for (let i = 0; i < presets.length; i++) {
      const p = presets[i]
      const t2 = Date.now()
      logStep(logs, `compose_${s.name}_${p.id}`, "start", `Composing ${p.name} for ${s.width}x${s.height}`)
      const compPayload: any = {
        copy: v0,
        analysis: analysis || {},
        crop_info: { width: s.width, height: s.height },
        smart_layout: false,
        variant: p.variant,
        ...(text_color_override ? { text_color_override } : {}),
      }
      const comp = await fetchJson(
        `${PIPELINE_URL}/compose`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(compPayload),
        },
        TIMEOUTS.compose,
      )
      comps.push({ ...comp, preset_id: p.id, preset_name: p.name, variant_used: p.variant })
      if (!composition) composition = comp // capture one for backward compat
      logStep(logs, `compose_${s.name}_${p.id}`, "ok", `Composed 1 variant`, undefined, Date.now() - t2)
    }

    // Render each composed SVG (exactly three)
    for (let i = 0; i < comps.length; i++) {
      const comp = comps[i]
      const label = comp?.preset_name || comp?.preset_id || `variant_${i + 1}`
      const t3 = Date.now()
      logStep(logs, `render_${s.name}_${label}`, "start", `Rendering ${s.width}x${s.height} for ${label}`)
      const renderPayload = {
        composition: comp,
        crop_info: { width: s.width, height: s.height },
        job_id,
        force_svg: true,
        analysis: analysis || {},
        use_gpt_image: false,
        copy: v0,
      }
      const r = await fetchJson(
        `${PIPELINE_URL}/render`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(renderPayload),
        },
        TIMEOUTS.render,
      )
      const outs = (r?.outputs || []).map((o: any) => ({ ...o, variant: label, size: s.name }))
      if (outs.length > 0) combinedOutputs.push(outs[0])
      if (!combinedThumb) combinedThumb = r?.thumbnail_url
      logStep(logs, `render_${s.name}_${label}`, "ok", "Rendered outputs", { outputs: outs.length }, Date.now() - t3)
    }

    renderRes = { outputs: combinedOutputs, thumbnail_url: combinedThumb }
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
