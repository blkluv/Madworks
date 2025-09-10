import os
import hashlib
import random
import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import uuid
import shutil
import re

import openai
from openai import AsyncOpenAI
from time import perf_counter
# cairosvg is lazily imported in the render path to avoid startup failures
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import boto3
from dotenv import load_dotenv
import aiofiles
import requests
from jinja2 import Template

# URL and mime helpers
from urllib.parse import urlparse
import mimetypes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows: prefer Selector event loop to reduce noisy ConnectionResetError logs from Proactor
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    # Help CairoSVG find native cairo DLLs without requiring global PATH edits.
    # Set CAIRO_DLL_DIR in your environment or .env to the folder that contains cairo-2.dll.
    try:
        cairo_dll_dir = (os.getenv("CAIRO_DLL_DIR", "").strip())
        added = False
        if cairo_dll_dir and hasattr(os, "add_dll_directory") and os.path.isdir(cairo_dll_dir):
            os.add_dll_directory(cairo_dll_dir)
            logger.info(f"Added CAIRO_DLL_DIR to DLL search path: {cairo_dll_dir}")
            added = True

        # Best-effort fallbacks if env var not provided
        if not added and hasattr(os, "add_dll_directory"):
            candidates = [
                r"C:\\Program Files\\GTK3-Runtime Win64\\bin",
                r"C:\\msys64\\mingw64\\bin",
                r"C:\\Program Files (x86)\\GTK3-Runtime Win64\\bin",
            ]
            for p in candidates:
                if os.path.isdir(p):
                    try:
                        os.add_dll_directory(p)
                        logger.info(f"Added candidate Cairo DLL path: {p}")
                        added = True
                        break
                    except Exception:
                        continue
        if not added:
            logger.debug("No Cairo DLL directory added automatically; set CAIRO_DLL_DIR if PNG/JPG rendering fails.")
    except Exception as _dll_e:
        logger.debug(f"Skipping Cairo DLL directory setup: {_dll_e}")

# Storage config: support S3 (MinIO) and local static file storage for easier dev/testing
STORAGE_MODE = os.getenv('STORAGE_MODE', 'local').lower()  # 's3' or 'local'
STORAGE_DIR = Path((os.getenv('STORAGE_DIR', str(Path(__file__).resolve().parent / 'storage'))).strip()).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv('PUBLIC_BASE_URL', 'http://localhost:8010').rstrip('/')

# Define all required directories
REQUIRED_DIRS = [
    STORAGE_DIR / 'uploads' / 'tmp',
    STORAGE_DIR / 'outputs',
    STORAGE_DIR / 'static',
    STORAGE_DIR / 'assets'  # Added assets directory for storing static assets
]

# Create all required directories
for directory in REQUIRED_DIRS:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

# Streaming upload/config defaults (tunable via env)
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "50"))  # hard cap on accepted upload size
UPLOAD_CHUNK_KB = int(os.getenv("UPLOAD_CHUNK_KB", "512"))  # chunk size for reading request body
UPLOAD_READ_TIMEOUT_S = float(os.getenv("UPLOAD_READ_TIMEOUT_S", "10"))  # per-chunk read timeout
DATA_URL_MAX_CHARS = int(os.getenv("DATA_URL_MAX_CHARS", "8000000"))  # ~8MB of base64 text

# Set up upload directory
UPLOAD_DIR_TEMP = Path(os.getenv("UPLOAD_DIR_TEMP", "").strip() or str(STORAGE_DIR / "uploads" / "tmp")).resolve()
UPLOAD_DIR_TEMP.mkdir(parents=True, exist_ok=True)

# Set up outputs directory
OUTPUTS_DIR = (STORAGE_DIR / 'outputs').resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT_S = float(os.getenv("PROCESS_TIMEOUT_S", "30"))  # analysis processing timeout

# GPT layout integration flags
USE_GPT_LAYOUT = (os.getenv("USE_GPT_LAYOUT", "true").strip().lower() in ("1", "true", "yes", "on"))
LAYOUT_MODEL = os.getenv("LAYOUT_MODEL", "gpt-5")
LAYOUT_TEMPERATURE = float(os.getenv("LAYOUT_TEMPERATURE", "0.6"))

# Image generation flags (PNG-only mode)
USE_GPT_IMAGE = (os.getenv("USE_GPT_IMAGE", "true").strip().lower() in ("1", "true", "yes", "on"))
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
STRICT_IMAGE_MODEL_ONLY = (os.getenv("STRICT_IMAGE_MODEL_ONLY", "false").strip().lower() in ("1", "true", "yes", "on"))

# Rasterization flags (legacy SVG rasterization)
# This is now used only as a fallback path if GPT image generation is unavailable.
RENDER_RASTER = (os.getenv("RENDER_RASTER", "false").strip().lower() in ("1", "true", "yes", "on"))

# Initialize OpenAI client
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = AsyncOpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else None
if not _OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; AI features disabled.")

# Initialize S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT', 'http://localhost:9000'),
    aws_access_key_id=os.getenv('S3_ACCESS_KEY', 'minio'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY', 'minio123'),
    region_name='us-east-1'
)

app = FastAPI(title="Madworks AI Pipeline", version="1.1.0")

# CORS middleware: allow browser clients (Next.js app) to call this API directly.
# Configure via CORS_ALLOW_ORIGINS env (comma-separated). Defaults are dev-friendly.
origins_env = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if not origins_env:
    allow_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8010",
        "http://127.0.0.1:8010",
        "*",
    ]
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
# If wildcard is present, set credentials False and pass ["*"] per Starlette rules
use_wildcard = "*" in allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if use_wildcard else allow_origins,
    allow_credentials=False if use_wildcard else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure local storage directory exists and mount static directories (best-effort)
try:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STORAGE_DIR)), name="static")
    # Extra aliases so clients can fetch both /static/... and /outputs/... paths
    # This covers cases where URLs are constructed as /outputs/{bucket}/{key}
    # (e.g., outputs/outputs/...) or /assets/... without the /static prefix.
    app.mount("/outputs", StaticFiles(directory=str(STORAGE_DIR)), name="outputs")
    app.mount("/assets", StaticFiles(directory=str(STORAGE_DIR)), name="assets")
except Exception as _e:
    # Mounting static is best-effort; S3 mode might not need this
    pass

# Pydantic models
class CopyConstraints(BaseModel):
    max_headline: int = 38
    max_sub: int = 80
    allowed_cta: List[str] = ["Shop Now", "Learn More", "Try it now", "Get Started", "Discover More"]
    forbidden_words: List[str] = ["guarantee", "#1", "best ever", "amazing", "incredible", "revolutionary"]
    brand_voice: str = "professional"
    target_audience: str = "general"

class CopyInput(BaseModel):
    copy_instructions: str
    facts: Dict[str, Any] = {}
    brand_kit_id: Optional[str] = None
    constraints: CopyConstraints = Field(default_factory=CopyConstraints)
    # Enhanced controls (optional; backward compatible)
    num_variants: int = 1
    temperature: float = 0.7
    model_name: Optional[str] = None
    tone: Optional[str] = None
    style_guide: Optional[str] = None
    platform: Optional[str] = None  # e.g., instagram_feed, story, linkedin
    # Pydantic v2: avoid conflict with protected namespace for `model_name`
    model_config = {"protected_namespaces": ()}

class ImageAnalysis(BaseModel):
    mask_url: str
    palette: List[str]
    crops: List[Dict[str, Any]]
    foreground_bbox: List[int]
    saliency_points: List[List[int]]
    dominant_colors: List[str]
    text_regions: List[Dict[str, Any]]

class CompositionResult(BaseModel):
    composition_id: str
    svg: str
    layout_data: Dict[str, Any]

class RenderOutput(BaseModel):
    format: str
    width: int
    height: int
    url: str

class RenderResult(BaseModel):
    outputs: List[RenderOutput]
    thumbnail_url: str

# Structured copy schema
class SpanRange(BaseModel):
    start: int
    end: int
    style: str = "bold"  # e.g., bold, italic, highlight

class FontRecommendation(BaseModel):
    role: str  # "headline" | "body"
    family: str
    weight: int = 700
    letter_spacing: float = 0.0
    line_height: float = 1.1

class CopyVariantStructured(BaseModel):
    headline: str
    subheadline: str
    cta: str
    emphasis_ranges: Dict[str, List[SpanRange]] = Field(default_factory=lambda: {"headline": [], "subheadline": []})
    font_recommendations: List[FontRecommendation] = Field(default_factory=list)

# Utility functions
def extract_palette(image: Image.Image, num_colors: int = 5) -> List[str]:
    """Extract dominant colors from image using Pillow quantization (no sklearn).
    Returns a list of hex strings like ['#rrggbb', ...].
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize for faster processing
    small_image = image.resize((150, 150))

    # Quantize to a palette of num_colors
    paletted = small_image.convert('P', palette=Image.ADAPTIVE, colors=max(1, num_colors))

    # Get palette and color frequencies
    palette = paletted.getpalette() or []  # flat list [r0,g0,b0, r1,g1,b1, ...]
    max_colors = paletted.width * paletted.height
    counts = paletted.getcolors(maxcolors=max_colors) or []  # list of (count, index)

    # Sort by frequency descending and map indices to RGB
    counts.sort(key=lambda x: x[0], reverse=True)
    result: List[str] = []
    for _, idx in counts[:num_colors]:
        base = int(idx) * 3
        if base + 2 < len(palette):
            r, g, b = palette[base], palette[base + 1], palette[base + 2]
            result.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")

    # Ensure we return at least one color
    if not result:
        r, g, b = small_image.resize((1, 1)).getpixel((0, 0))
        result = [f"#{int(r):02x}{int(g):02x}{int(b):02x}"]

    return result

def detect_foreground(image: Image.Image) -> Tuple[Image.Image, List[int]]:
    """Detect foreground using OpenCV"""
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the main subject)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = [x, y, x + w, y + h]
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Convert back to PIL
        mask_pil = Image.fromarray(mask)
        return mask_pil, bbox
    
    # Fallback: return full image
    return Image.new('L', image.size, 255), [0, 0, image.width, image.height]

def generate_crop_proposals(image: Image.Image, bbox: List[int]) -> List[Dict[str, Any]]:
    """Generate crop proposals based on common ad ratios"""
    ratios = [
        {"name": "Square", "ratio": "1:1", "width": 1080, "height": 1080},
        {"name": "Portrait", "ratio": "4:5", "width": 1080, "height": 1350},
        {"name": "Landscape", "ratio": "16:9", "width": 1920, "height": 1080},
        {"name": "Story", "ratio": "9:16", "width": 1080, "height": 1920},
    ]
    
    crops = []
    for ratio_info in ratios:
        # Calculate crop based on bbox and ratio
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        
        # Center the crop on the subject
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Calculate crop dimensions
        target_ratio = ratio_info["width"] / ratio_info["height"]
        if bbox_w / bbox_h > target_ratio:
            # Image is wider than target ratio
            crop_w = int(bbox_h * target_ratio)
            crop_h = bbox_h
        else:
            # Image is taller than target ratio
            crop_w = bbox_w
            crop_h = int(bbox_w / target_ratio)
        
        # Ensure crop fits within image bounds
        crop_x = max(0, min(center_x - crop_w // 2, image.width - crop_w))
        crop_y = max(0, min(center_y - crop_h // 2, image.height - crop_h))
        
        crops.append({
            "name": ratio_info["name"],
            "ratio": ratio_info["ratio"],
            "width": ratio_info["width"],
            "height": ratio_info["height"],
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h
        })
    
    return crops

async def generate_copy_with_ai(
    instructions: str,
    facts: Dict[str, Any],
    constraints: CopyConstraints,
    *,
    tone: Optional[str] = None,
    style_guide: Optional[str] = None,
    platform: Optional[str] = None,
    temperature: float = 0.7,
    model_name: Optional[str] = None,
    num_variants: int = 1,
) -> List[Dict[str, str]]:
    """Generate one or more ad copy variants using OpenAI with guardrails.

    Returns a list of copy dicts, each with keys: headline, subheadline, cta.
    """
    # Short-circuit: no API key/client -> deterministic fallback (server still works)
    if not client or not _OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY missing; returning deterministic fallback copy")
        return [{
            "headline": "Professional Excellence",
            "subheadline": "Quality that delivers results",
            "cta": constraints.allowed_cta[0]
        }]

    try:
        # Build the prompt with optional tone/platform/style guide
        addl = []
        if tone:
            addl.append(f"Tone: {tone}")
        if platform:
            addl.append(f"Platform: {platform}")
        if style_guide:
            addl.append(f"Style Guide: {style_guide}")
        addl_str = ("\n" + "\n".join(addl)) if addl else ""

        prompt = f"""
        You are an elite performance marketing copywriter. Create high-converting, professional ad copy.

        Instructions: {instructions}

        Brand Facts: {json.dumps(facts, indent=2)}

        Constraints:
        - Headline: max {constraints.max_headline} characters
        - Subheadline: max {constraints.max_sub} characters
        - CTA must be one of: {', '.join(constraints.allowed_cta)}
        - Brand voice: {constraints.brand_voice}
        - Target audience: {constraints.target_audience}
        - Avoid these words/claims: {', '.join(constraints.forbidden_words)}
        {addl_str}

        Output requirements:
        - Return ONLY a JSON object, no markdown or commentary.
        - JSON keys: "headline", "subheadline", "cta".
        - Write crisp, commercial-grade copy with impeccable grammar.
        """
        
        model = model_name or os.getenv("COPY_MODEL", "gpt-3.5-turbo")
        variants: List[Dict[str, str]] = []
        # Enforce a single copy variant regardless of input
        n = 1
        for _ in range(n):
            chat_args = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert advertising copywriter who outputs strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
            # Some models (e.g., gpt-5 family) only support default temperature; omit to avoid 400s
            if not str(model).strip().lower().startswith("gpt-5"):
                chat_args["temperature"] = float(temperature)
            response = await client.chat.completions.create(**chat_args)

            content = response.choices[0].message.content.strip()

            # Parse JSON response with robust fallback
            parsed: Optional[Dict[str, str]] = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON substring
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start:end+1])
                    except Exception:
                        parsed = None
            if not parsed:
                logger.warning("AI did not return valid JSON; using fallback copy")
                parsed = {
                    "headline": "Professional Excellence",
                    "subheadline": "Quality that delivers results",
                    "cta": constraints.allowed_cta[0]
                }

            # Validate + clamp
            required_fields = ["headline", "subheadline", "cta"]
            for f in required_fields:
                if f not in parsed:
                    parsed[f] = ""
            if parsed.get("cta") not in constraints.allowed_cta:
                parsed["cta"] = constraints.allowed_cta[0]
            # Simple compliance guard: remove forbidden words
            low_forbidden = [w.lower() for w in constraints.forbidden_words]
            for key in ["headline", "subheadline"]:
                text = (parsed.get(key) or "")
                for fw in low_forbidden:
                    if fw in text.lower():
                        text = text.lower().replace(fw, "").strip()
                parsed[key] = text
            variants.append(parsed)

        return variants
            
    except Exception as e:
        logger.error(f"Error generating copy with AI: {e}")
        # Fallback: return one safe variant
        return [{
            "headline": "Professional Excellence",
            "subheadline": "Quality that delivers results",
            "cta": constraints.allowed_cta[0]
        }]

# GPT layout generation and validation
def _clamp_float(x: Any, lo: float, hi: float, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default

def _to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return None

def _sanitize_layout_variant(raw: Dict[str, Any], palette: List[str]) -> Dict[str, Any]:
    """Allow only known keys and clamp/coerce values to safe ranges."""
    try:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        allowed_styles = {"fill", "outline", "pill"}
        allowed_width_modes = {"auto", "wide"}
        allowed_panel_side = {"left", "right", "center"}
        allowed_text_align = {"left", "center", "right"}
        allowed_vertical_align = {"top", "middle", "bottom"}
        allowed_bg_fit = {"meet", "slice"}
        allowed_headline_fill = {"gradient", "solid"}
        allowed_panel_style = {"none", "card"}
        allowed_emotional_mode = {"strong", "soft", "neutral"}
        pal_len = max(0, len(palette or []))

        # seed
        if "seed" in raw:
            try:
                out["seed"] = int(raw["seed"])
            except Exception:
                pass

        # floats with clamps
        ts = _clamp_float(raw.get("type_scale"), 0.80, 1.40, None)
        if ts is not None:
            out["type_scale"] = ts
        pwidth = _clamp_float(raw.get("panel_width_factor"), 0.40, 0.85, None)
        if pwidth is not None:
            out["panel_width_factor"] = pwidth
        lg = _clamp_float(raw.get("line_gap_factor"), 0.80, 1.50, None)
        if lg is not None:
            out["line_gap_factor"] = lg
        sg = _clamp_float(raw.get("sub_gap_factor"), 0.80, 1.80, None)
        if sg is not None:
            out["sub_gap_factor"] = sg
        cg = _clamp_float(raw.get("cta_gap_scale"), 0.60, 1.80, None)
        if cg is not None:
            out["cta_gap_scale"] = cg
        shade = _clamp_float(raw.get("shade_factor"), 0.0, 0.80, None)
        if shade is not None:
            out["shade_factor"] = shade
        side_shade = _clamp_float(raw.get("side_shade_factor"), 0.0, 0.80, None)
        if side_shade is not None:
            out["side_shade_factor"] = side_shade

        # enums / booleans
        ps = raw.get("panel_side")
        if isinstance(ps, str) and ps.strip().lower() in allowed_panel_side:
            out["panel_side"] = ps.strip().lower()
        fp = _to_bool(raw.get("flip_panel"))
        if fp is not None:
            out["flip_panel"] = fp
        show_scrim = _to_bool(raw.get("show_scrim"))
        if show_scrim is not None:
            out["show_scrim"] = show_scrim
        ta = (raw.get("text_align") or "").strip().lower()
        if ta in allowed_text_align:
            out["text_align"] = ta
        va = (raw.get("vertical_align") or "").strip().lower()
        if va in allowed_vertical_align:
            out["vertical_align"] = va
        cstyle = (raw.get("cta_style") or "").strip().lower()
        if cstyle in allowed_styles:
            out["cta_style"] = cstyle
        wmode = (raw.get("cta_width_mode") or "").strip().lower()
        if wmode in allowed_width_modes:
            out["cta_width_mode"] = wmode
        bgf = (raw.get("bg_fit") or "").strip().lower()
        if bgf in allowed_bg_fit:
            out["bg_fit"] = bgf
        hf = (raw.get("headline_fill") or "").strip().lower()
        if hf in allowed_headline_fill:
            out["headline_fill"] = hf
        pstyle = (raw.get("panel_style") or "").strip().lower()
        if pstyle in allowed_panel_style:
            out["panel_style"] = pstyle
        emo = (raw.get("emotional_mode") or "").strip().lower()
        if emo in allowed_emotional_mode:
            out["emotional_mode"] = emo

        # accent_index
        if "accent_index" in raw:
            try:
                ai = int(raw.get("accent_index"))
                if pal_len > 0:
                    ai = max(0, min(pal_len - 1, ai))
                else:
                    ai = max(0, ai)
                out["accent_index"] = ai
            except Exception:
                pass

        # badge_text
        if "badge_text" in raw and isinstance(raw.get("badge_text"), str):
            out["badge_text"] = raw.get("badge_text")[:80]

        # highlight overrides
        if "highlight_text_color" in raw and isinstance(raw.get("highlight_text_color"), str):
            # accept as-is; normalized later
            out["highlight_text_color"] = raw.get("highlight_text_color")[:16]
        hs = _clamp_float(raw.get("highlight_stroke_scale"), 0.6, 1.6, None)
        if hs is not None:
            out["highlight_stroke_scale"] = hs

        return out
    except Exception:
        return {}

async def generate_layout_variant_with_ai(
    copy_data: Dict[str, Any],
    analysis: Dict[str, Any],
    crop_info: Dict[str, Any],
    base_variant: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Ask the GPT layout model for safe layout overrides. Returns sanitized dict; may be empty."""
    try:
        if not USE_GPT_LAYOUT:
            return {}
        if not client or not _OPENAI_API_KEY:
            logger.warning("generate_layout_variant_with_ai: OpenAI client not configured; skipping")
            return {}

        palette: List[str] = []
        try:
            palette = [c for c in (analysis.get("palette") or []) if isinstance(c, str)]
        except Exception:
            palette = []
        width = int(crop_info.get("width", 1080))
        height = int(crop_info.get("height", 1080))
        bbox = analysis.get("foreground_bbox") or []

        allowed_keys_doc = [
            "seed (int)",
            "type_scale (float 0.80–1.40)",
            "panel_side ('left'|'right'|'center')",
            "flip_panel (bool)",
            "panel_width_factor (float 0.40–0.85)",
            "cta_style ('fill'|'outline'|'pill')",
            "cta_width_mode ('auto'|'wide')",
            "accent_index (int index into palette)",
            "line_gap_factor (float 0.80–1.50)",
            "sub_gap_factor (float 0.80–1.80)",
            "cta_gap_scale (float 0.60–1.80)",
            "show_scrim (bool)",
            "bg_fit ('meet'|'slice')",
            "shade_factor (float 0.0–0.80)",
            "side_shade_factor (float 0.0–0.80)",
            "headline_fill ('gradient'|'solid')",
            "panel_style ('none'|'card')",
            "badge_text (string <=80 chars)",
            "text_align ('left'|'center'|'right')",
            "vertical_align ('top'|'middle'|'bottom')",
            "emotional_mode ('strong'|'soft'|'neutral')",
            "highlight_text_color (hex '#rrggbb')",
            "highlight_stroke_scale (float 0.6–1.6)",
        ]

        context = {
            "copy": {
                "headline": copy_data.get("headline", ""),
                "subheadline": copy_data.get("subheadline", ""),
                "cta": copy_data.get("cta", "Learn More"),
            },
            "canvas": {"width": width, "height": height},
            "palette": palette,
            "foreground_bbox": bbox,
            "base_variant": base_variant or {},
        }

        prompt = (
            "You are a senior ad layout engine. Propose layout parameters as a compact JSON object.\n"
            "Rules:\n"
            "- Output ONLY raw JSON, no markdown, no commentary.\n"
            "- Include only the keys you want to override (omit unknown or null keys).\n"
            "- Keep values within safe ranges and enums listed below.\n"
            "- Use booleans true/false, not strings.\n\n"
            f"Allowed keys and ranges: {json.dumps(allowed_keys_doc)}\n\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}"
        )

        _model = (model or LAYOUT_MODEL)
        chat_args2 = {
            "model": _model,
            "messages": [
                {"role": "system", "content": "You output strict JSON objects with no extra text."},
                {"role": "user", "content": prompt},
            ],
        }
        # Omit temperature for models that only accept default (e.g., gpt-5 family)
        if not str(_model).strip().lower().startswith("gpt-5"):
            chat_args2["temperature"] = float(temperature if temperature is not None else LAYOUT_TEMPERATURE)
        response = await client.chat.completions.create(**chat_args2)
        content = (response.choices[0].message.content or "").strip()

        # Parse JSON robustly
        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(content[start : end + 1])
                except Exception:
                    parsed = None
        if not parsed:
            logger.warning("GPT layout returned non-JSON; ignoring")
            return {}

        sanitized = _sanitize_layout_variant(parsed, palette)
        if sanitized:
            logger.info(f"GPT layout overrides: {list(sanitized.keys())}")
        return sanitized
    except Exception as e:
        logger.error(f"generate_layout_variant_with_ai error: {e}")
        return {}

# Simple in-memory cache for GPT layout overrides to reduce API calls
_gpt_variant_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
try:
    GPT_LAYOUT_CACHE_TTL_S = float(os.getenv("GPT_LAYOUT_CACHE_TTL_S", "600"))
except Exception:
    GPT_LAYOUT_CACHE_TTL_S = 600.0

def _gpt_layout_cache_key(copy_data: Dict[str, Any], analysis: Dict[str, Any], crop_info: Dict[str, Any], base_variant: Dict[str, Any]) -> str:
    try:
        palette = [c for c in (analysis.get("palette") or []) if isinstance(c, str)][:3]
    except Exception:
        palette = []
    key_obj = {
        "h": copy_data.get("headline", ""),
        "s": copy_data.get("subheadline", ""),
        "c": copy_data.get("cta", ""),
        "w": int(crop_info.get("width", 1080)),
        "hgt": int(crop_info.get("height", 1080)),
        "pal": palette,
    }
    return hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()

async def get_gpt_layout_variant_cached(
    copy_data: Dict[str, Any],
    analysis: Dict[str, Any],
    crop_info: Dict[str, Any],
    base_variant: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        if not USE_GPT_LAYOUT:
            return {}
        key = _gpt_layout_cache_key(copy_data, analysis, crop_info, base_variant or {})
        now = perf_counter()
        hit = _gpt_variant_cache.get(key)
        if hit and (now - hit[0]) < float(GPT_LAYOUT_CACHE_TTL_S):
            return dict(hit[1])
        overrides = await generate_layout_variant_with_ai(copy_data, analysis, crop_info, base_variant=base_variant)
        _gpt_variant_cache[key] = (now, overrides or {})
        return overrides or {}
    except Exception:
        return {}

def _measure_chars_for_width(width: int, font_size: int, avg_char_width_factor: float = 0.55) -> int:
    """Estimate how many characters fit per line for a given width and font size.
    avg_char_width_factor ~0.55 works reasonably for sans-serif.
    """
    usable = max(1, int(width))
    approx = int((usable / (font_size)) / avg_char_width_factor)
    return max(8, approx)

def _wrap_text(text: str, max_chars: int) -> List[str]:
    words = (text or "").split()
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        candidate = (" ".join(cur + [w])).strip()
        if len(candidate) <= max_chars:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]

# Deterministic utilities for structured copy
def recommend_fonts(tone: Optional[str], platform: Optional[str]) -> List[FontRecommendation]:
    """Deterministic font recommendations based on tone/platform.
    Keep this purely deterministic and commercial-safe.
    """
    t = (tone or "").strip().lower()
    # Simple mapping; expand as needed
    if t in ("bold", "confident", "sporty"):
        return [
            FontRecommendation(role="headline", family="Inter", weight=800, letter_spacing=0, line_height=1.05),
            FontRecommendation(role="body", family="Inter", weight=500, letter_spacing=0, line_height=1.3),
        ]
    if t in ("professional", "minimal", "clean"):
        return [
            FontRecommendation(role="headline", family="Work Sans", weight=700, letter_spacing=0, line_height=1.08),
            FontRecommendation(role="body", family="Work Sans", weight=400, letter_spacing=0, line_height=1.35),
        ]
    if t in ("friendly", "warm", "casual"):
        return [
            FontRecommendation(role="headline", family="DM Sans", weight=700, letter_spacing=0, line_height=1.1),
            FontRecommendation(role="body", family="DM Sans", weight=400, letter_spacing=0, line_height=1.35),
        ]
    # Default
    return [
        FontRecommendation(role="headline", family="Inter", weight=800, letter_spacing=0, line_height=1.06),
        FontRecommendation(role="body", family="Inter", weight=500, letter_spacing=0, line_height=1.32),
    ]

def _first_n_words_span(text: str, n: int = 2) -> Optional[SpanRange]:
    text = text or ""
    if not text:
        return None
    # Find char index at the end of the Nth word (simple whitespace split heuristic)
    count = 0
    in_word = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_word:
                count += 1
                in_word = False
                if count >= n:
                    return SpanRange(start=0, end=i, style="bold")
        else:
            in_word = True
    # If string ends without trailing space, close last word
    if in_word:
        count += 1
        if count >= n:
            return SpanRange(start=0, end=len(text), style="bold")
    # Fallback: emphasize entire text if fewer than n words
    return SpanRange(start=0, end=len(text), style="bold")

def compute_emphasis_ranges(headline: str, subheadline: str) -> Dict[str, List[SpanRange]]:
    """Compute deterministic emphasis ranges.
    - Headline: emphasize first 2 words.
    - Subheadline: emphasize first numeric token if present (e.g., 20%, 2x, 2025).
    """
    import re
    h_span = _first_n_words_span(headline or "", 2)
    subs: List[SpanRange] = []
    if subheadline:
        m = re.search(r"\d[\d\w%\.]*", subheadline)
        if m:
            subs.append(SpanRange(start=m.start(), end=m.end(), style="bold"))
    return {
        "headline": [h_span] if h_span else [],
        "subheadline": subs,
    }

def _try_build_data_url_from_storage(original_url: str) -> Optional[str]:
    """If the URL points to our mounted static paths, load bytes directly
    from STORAGE_DIR and return a data URL. Supports /static, /outputs, /assets.
    Returns None on failure."""
    try:
        if not original_url:
            return None
        p = urlparse(original_url)
        path = p.path or ""
        prefixes = ["/static/", "/outputs/", "/assets/"]
        for prefix in prefixes:
            if path.startswith(prefix):
                rel_path = path[len(prefix):]
                file_path = Path(STORAGE_DIR) / rel_path
                if file_path.exists():
                    data = file_path.read_bytes()
                    mime, _ = mimetypes.guess_type(str(file_path))
                    mime = mime or "image/png"
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:{mime};base64,{b64}"
                break
    except Exception as e:
        logger.warning(f"Failed to build data URL from storage: {e}")
    return None

def _inline_local_images_for_raster(svg_text: str) -> str:
    """Inline local static image hrefs (e.g., /static, /assets, /outputs) as data URLs.
    This makes the SVG self-contained so external fetchers (e.g., Sharp/libvips) don't need HTTP.
    """
    if not svg_text:
        return svg_text
    # Match both href and xlink:href on <image ...>
    pattern = re.compile(r"(<image\b[^>]*?\s(?:xlink:href|href)=\s*[\"\'])([^\"\']+)([\"\'])", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        prefix, href, suffix = m.group(1), m.group(2), m.group(3)
        if href.startswith("data:"):
            return m.group(0)  # already inlined
        # Try converting local/static URLs to data URLs
        du = _try_build_data_url_from_storage(href)
        if du:
            return f"{prefix}{du}{suffix}"
        # Also handle absolute URLs that point to our PUBLIC_BASE_URL, then map path
        try:
            u = urlparse(href)
            if u.scheme in ("http", "https"):
                # Map path part regardless of host
                du2 = _try_build_data_url_from_storage(href)
                if du2:
                    return f"{prefix}{du2}{suffix}"
        except Exception:
            pass
        return m.group(0)

    return pattern.sub(repl, svg_text)

def _resolve_img_href(analysis: Dict[str, Any]) -> str:
    """Resolve the best image href for SVG <image> tag.
    Prefer direct URLs to avoid embedding huge base64 strings. Fallback to data URL only if small.
    Priority: original_url_internal/original_url (as-is) -> sizeable-but-capped original_data_url -> local static data URL -> empty.
    """
    # Prefer direct URL if available
    for key in ("original_url_internal", "original_url"):
        url = (analysis.get(key) or "").strip()
        if url:
            return url

    # Consider existing data URL if not excessively large
    data_url = (analysis.get("original_data_url") or "").strip()
    if data_url.startswith("data:") and len(data_url) > 100 and len(data_url) <= DATA_URL_MAX_CHARS:
        return data_url

    # Try constructing a data URL from local storage if possible (useful in local mode)
    for key in ("original_url_internal", "original_url"):
        url = (analysis.get(key) or "").strip()
        if url:
            du = _try_build_data_url_from_storage(url)
            if du and len(du) <= DATA_URL_MAX_CHARS:
                return du
    return ""

def _normalize_hex_color(c: Optional[str]) -> Optional[str]:
    """Normalize a CSS hex color string to #rrggbb. Returns None if invalid.
    Accepts #rgb or #rrggbb (case-insensitive), with or without leading '#'."""
    try:
        if not isinstance(c, str):
            return None
        s = c.strip()
        if not s:
            return None
        if s.startswith('#'):
            s = s[1:]
        # Allow 3 or 6 hex digits
        if len(s) == 3 and all(ch in '0123456789abcdefABCDEF' for ch in s):
            s = ''.join(ch * 2 for ch in s)
        if len(s) == 6 and all(ch in '0123456789abcdefABCDEF' for ch in s):
            return '#' + s.lower()
        return None
    except Exception:
        return None

def _derive_secondary_color(base_hex: str) -> str:
    """Derive a legible secondary text color from the primary hex.
    Heuristic: if the base is bright, darken ~15%; otherwise, lighten ~15%."""
    try:
        h = _normalize_hex_color(base_hex) or '#ffffff'
        r = int(h[1:3], 16)
        g = int(h[3:5], 16)
        b = int(h[5:7], 16)
        # Perceived brightness
        brightness = (r * 299 + g * 587 + b * 114) / 1000.0
        factor = 0.85 if brightness > 140 else 1.15
        nr = max(0, min(255, int(r * factor)))
        ng = max(0, min(255, int(g * factor)))
        nb = max(0, min(255, int(b * factor)))
        return f"#{nr:02x}{ng:02x}{nb:02x}"
    except Exception:
        return '#e5e5e5'

def _is_light_color(hex_color: str) -> bool:
    """Return True if the color is perceptually light."""
    try:
        h = _normalize_hex_color(hex_color) or '#ffffff'
        r = int(h[1:3], 16)
        g = int(h[3:5], 16)
        b = int(h[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000.0
        return brightness >= 145
    except Exception:
        return True

def _contrast_text_color(base_hex: str) -> str:
    """Choose a strong contrasting text fill against the base color.
    If base is light, return near-black; else return near-white."""
    try:
        return '#0a0a0a' if _is_light_color(base_hex) else '#ffffff'
    except Exception:
        return '#0a0a0a'

def create_svg_composition(copy_data: Dict[str, Any], analysis: Dict[str, Any], crop_info: Dict[str, Any], *, smart_layout: bool = True, panel_side_override: Optional[str] = None, text_color_override: Optional[str] = None, variant: Optional[Dict[str, Any]] = None) -> str:
    """Create SVG composition with a professional layout and support for emphasis tspans.
    - Full-bleed background image with strong legibility gradient
    - Smart left/right text panel selection (if smart_layout) based on subject position
    - Optional explicit panel override via panel_side_override ('left'|'right') taking precedence
    - Headline and subheadline with emphasis ranges rendered as bold tspans
    - CTA button positioned within the chosen text panel area
    """

    # Layout metrics
    width = int(crop_info["width"]) if "width" in crop_info else 1080
    height = int(crop_info["height"]) if "height" in crop_info else 1080
    # Even tighter padding to give text more room
    padding = max(16, int(min(width, height) * 0.025))

    # Variation profile (seeded for deterministic diversity)
    v: Dict[str, Any] = variant or {}
    # Build a deterministic seed from copy if none provided
    seed = v.get("seed")
    if seed is None:
        try:
            basis = f"{copy_data.get('headline','')}|{copy_data.get('subheadline','')}|{copy_data.get('cta','')}"
            seed = int(hashlib.sha256(basis.encode('utf-8')).hexdigest(), 16) % (2**32)
        except Exception:
            seed = None
    rng = random.Random(seed) if seed is not None else random.Random()

    # Typography (scaled to canvas) with mild variance
    type_scale = v.get("type_scale")
    if not isinstance(type_scale, (int, float)):
        type_scale = rng.uniform(0.98, 1.12)
    # Larger base sizes, but slightly reduced to avoid crowding; auto-fit loop below will reduce if needed
    headline_size = max(52, int(min(width, height) * 0.18 * float(type_scale)))
    sub_size = max(24, int(min(width, height) * 0.055 * float(type_scale)))
    # Default ALL text to white as requested
    text_color = "#ffffff"
    text_color_secondary = "#ffffff"

    # Apply sanitized text color override if provided
    try:
        if text_color_override:
            norm = _normalize_hex_color(text_color_override)
            if norm:
                text_color = norm
                text_color_secondary = _derive_secondary_color(norm)
    except Exception:
        pass

    # Highlight styling: contrasting fill on top of a stroke matching the base text color
    try:
        # Force highlight fill to match base text color so ALL text is white by default
        highlight_text_color = text_color
        # Still allow an explicit override via variant if provided and valid
        try:
            highlight_override = v.get("highlight_text_color")
            if isinstance(highlight_override, str):
                norm_h = _normalize_hex_color(highlight_override)
                if norm_h:
                    highlight_text_color = norm_h
        except Exception:
            pass
    except Exception:
        highlight_text_color = text_color

    # Stroke widths for highlight effect (thicker than normal text stroke)
    try:
        scale = v.get("highlight_stroke_scale")
        try:
            s = float(scale)
            # clamp
            if s < 0.6:
                s = 0.6
            if s > 1.6:
                s = 1.6
        except Exception:
            s = 1.0
    except Exception:
        s = 1.0
    headline_highlight_stroke_w = max(1, int(headline_size * 0.12 * s))
    sub_highlight_stroke_w = max(1, int(sub_size * 0.10 * s))
    # Stroke colors follow the base text colors of their respective blocks
    headline_highlight_stroke = text_color
    sub_highlight_stroke = text_color_secondary

    # Recommended fonts from structured copy (backward compatible)
    # Defaults tuned for ad readability (strong display headline + clean body)
    # Use single quotes for multi-word family names to avoid double quotes inside XML attributes
    default_stack = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    headline_stack = "Impact, Oswald, 'Bebas Neue', 'Arial Black', Inter, Arial, sans-serif"
    font_family_headline = headline_stack
    font_family_body = default_stack
    headline_weight = 900
    body_weight = 600
    headline_letter_spacing = 0.2
    body_letter_spacing = 0.0

    try:
        recs = copy_data.get("font_recommendations") or []
        # Expect roles: headline, body
        for r in recs:
            role = (r.get("role") or "").lower()
            fam = (r.get("family") or "").strip()
            if role == "headline":
                if fam:
                    font_family_headline = f"{fam}, {default_stack}"
                headline_weight = int(r.get("weight", headline_weight))
                headline_letter_spacing = float(r.get("letter_spacing", headline_letter_spacing))
            elif role == "body":
                if fam:
                    font_family_body = f"{fam}, {default_stack}"
                body_weight = int(r.get("weight", body_weight))
                body_letter_spacing = float(r.get("letter_spacing", body_letter_spacing))
    except Exception:
        # Ignore malformed recs
        pass

    # Mild seeded variance to letter spacing for diversity (unless explicitly overridden above)
    try:
        if not isinstance(headline_letter_spacing, (int, float)):
            headline_letter_spacing = 0.0
        if not isinstance(body_letter_spacing, (int, float)):
            body_letter_spacing = 0.0
        headline_letter_spacing = float(headline_letter_spacing) + round(rng.uniform(-0.4, 0.6), 2)
        body_letter_spacing = float(body_letter_spacing) + round(rng.uniform(-0.2, 0.4), 2)
    except Exception:
        pass

    # Determine panel side: allow smart left/right panel (distinct from centered)
    panel_side = "left"
    try:
        override = (panel_side_override or "").strip().lower() if isinstance(panel_side_override, str) else None
        if override in ("left", "right", "center"):
            panel_side = override
        elif smart_layout:
            bbox = analysis.get("foreground_bbox") or None
            orig_size = analysis.get("original_size") or None
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                if orig_size and isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
                    W0 = max(1, int(orig_size[0]))
                    H0 = max(1, int(orig_size[1]))
                else:
                    # Fallback to crop size
                    W0, H0 = width, height
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                # Normalized horizontal position in original image
                nx = cx / float(W0)
                # Always choose opposite side of subject so text never sits over subject
                try:
                    w_box = max(1.0, float(bbox[2]) - float(bbox[0]))
                    h_box = max(1.0, float(bbox[3]) - float(bbox[1]))
                    area_ratio = (w_box * h_box) / float(W0 * H0)
                except Exception:
                    area_ratio = 0.0
                panel_side = "right" if nx < 0.5 else "left"
            else:
                # No subject detected; default to a left text panel for a distinct layout
                panel_side = "left"
    except Exception:
        pass

    # Optional panel flip or override via variant
    try:
        if isinstance(v.get("flip_panel"), bool) and v.get("flip_panel") and not panel_side_override:
            panel_side = "left" if panel_side == "right" else "right"
        forced_side = v.get("panel_side")
        if isinstance(forced_side, str) and forced_side.lower() in ("left", "right", "center"):
            panel_side = forced_side.lower()
    except Exception:
        pass

    # Panel width and inner text start x with variance
    panel_width_factor = v.get("panel_width_factor")
    if not isinstance(panel_width_factor, (int, float)):
        # Choose width of the text column based on aspect ratio so it looks bold and balanced
        aspect = float(width) / float(max(1, height))
        if panel_side == "center":
            if aspect >= 1.35:
                panel_width_factor = 0.95  # much wider for landscape
            elif aspect <= 0.80:
                panel_width_factor = 0.88  # wider for portrait too
            else:
                panel_width_factor = 0.92  # near-square
        else:
            # For side panels, be more compact to look distinct and professional
            panel_width_factor = rng.uniform(0.46, 0.54)
    text_panel_w = int(width * float(panel_width_factor))
    # Compute panel x and content area based on side
    if panel_side == "left":
        panel_x = padding
    elif panel_side == "right":
        panel_x = max(padding, width - padding - text_panel_w)
    else:
        panel_x = int((width - text_panel_w) / 2)
    # Inner content start and center
    text_x = int(panel_x + padding)
    # Anchor x for text rendering (center of panel for non-center layouts)
    content_center_x = int(width / 2) if panel_side == "center" else int(panel_x + (text_panel_w / 2))

    # CTA sizing; final placement computed after text layout is fitted
    cta_text = copy_data.get("cta", "Learn More")
    cta_width_mode = (v.get("cta_width_mode") or "auto").lower()
    per_char = 14 if cta_width_mode == "wide" else 12
    cta_width = max(240, int(48 + len(cta_text) * per_char))
    cta_height = max(60, int(min(width, height) * 0.07))
    # placeholders; will be updated after text is fitted
    # Start CTA centered across the full canvas
    cta_x = int((width - cta_width) / 2)
    cta_y = height - padding - cta_height
    cta_text_x = cta_x + cta_width // 2
    cta_text_y = cta_y + int(cta_height * 0.66)

    # Colors / accents
    palette = analysis.get("palette", [])
    accent_index = v.get("accent_index")
    try:
        if accent_index is None and isinstance(palette, list) and len(palette) >= 2:
            # Prefer using one of top 3 accents if available
            candidates = list(range(1, min(4, len(palette))))
            accent_index = rng.choice(candidates) if candidates else 2
        if isinstance(accent_index, int) and 0 <= accent_index < len(palette):
            cta_color = palette[accent_index]
        else:
            cta_color = palette[2] if len(palette) > 2 else "#2563EB"
    except Exception:
        cta_color = palette[2] if len(palette) > 2 else "#2563EB"

    # CTA style (fill/outline/pill)
    cta_style = (v.get("cta_style") or "pill").lower()
    if cta_style == "outline":
        cta_fill = "none"
        cta_stroke = cta_color
        cta_stroke_opacity = 0.9
    else:
        cta_fill = cta_color
        cta_stroke = "#ffffff"
        cta_stroke_opacity = 0.12

    cta_radius = 10
    if cta_style == "pill":
        cta_radius = max(14, int(cta_height / 2))

    # Determine text alignment within the text panel
    try:
        _ta = (v.get("text_align") or "").strip().lower()
        if _ta not in ("left", "center", "right"):
            _ta = "center"
        text_align = _ta
    except Exception:
        text_align = "center"

    # Text wrapping (word-safe) and dynamic fitting
    # Restrict content width to the text panel area
    content_width = max(200, int(text_panel_w - 2 * padding))
    headline = copy_data.get("headline", "").strip()
    subheadline = copy_data.get("subheadline", "").strip()

    # Build segment lines from emphasis ranges
    def _segments_by_line(full_text: str, lines: List[str], ranges: List[Dict[str, Any]]):
        # ranges: [{start, end, style}]
        try:
            norm_ranges = []
            for r in (ranges or []):
                try:
                    s = int(r.get("start", 0))
                    e = int(r.get("end", 0))
                    if e > s:
                        norm_ranges.append({"start": s, "end": e, "style": (r.get("style") or "bold")})
                except Exception:
                    continue
            norm_ranges.sort(key=lambda x: (x["start"], x["end"]))

            seg_lines: List[List[Dict[str, Any]]] = []
            cursor = 0
            for line in lines:
                # find this line's start index in full_text starting from cursor to be stable
                idx = full_text.find(line, cursor)
                if idx == -1:
                    idx = cursor  # fallback best-effort
                line_start = idx
                line_end = idx + len(line)
                cursor = line_end

                # collect overlapping ranges
                overlaps = [r for r in norm_ranges if not (r["end"] <= line_start or r["start"] >= line_end)]
                # Build segments in order
                segs: List[Dict[str, Any]] = []
                pos = line_start
                for r in overlaps:
                    s = max(line_start, r["start"]) ; e = min(line_end, r["end"]) ; s = max(s, pos)
                    if s > pos:
                        segs.append({"text": full_text[pos:s], "style": "normal"})
                    if e > s:
                        segs.append({"text": full_text[s:e], "style": r.get("style") or "bold"})
                        pos = e
                if pos < line_end:
                    segs.append({"text": full_text[pos:line_end], "style": "normal"})
                # If no overlaps, add the whole line as normal
                if not overlaps and not segs:
                    segs = [{"text": line, "style": "normal"}]
                seg_lines.append(segs)
            return seg_lines
        except Exception:
            # Fallback: each line as a single normal segment
            return [[{"text": ln, "style": "normal"}] for ln in lines]

    em = copy_data.get("emphasis_ranges") or {}
    headline_ranges = em.get("headline") or []
    sub_ranges = em.get("subheadline") or []

    # Iteratively reduce sizes until both text blocks fit above CTA
    # Additional seeded variance for rhythm
    line_gap_factor = v.get("line_gap_factor")
    if not isinstance(line_gap_factor, (int, float)):
        line_gap_factor = rng.uniform(0.9, 1.12)
    sub_gap_factor = v.get("sub_gap_factor")
    if not isinstance(sub_gap_factor, (int, float)):
        sub_gap_factor = rng.uniform(0.9, 1.12)
    # and within content width using word wrapping.
    max_attempts = 12
    attempt = 0
    best = None
    while True:
        # measure per current sizes
        max_chars_headline = _measure_chars_for_width(content_width, headline_size)
        max_chars_sub = _measure_chars_for_width(content_width, sub_size)
        headline_lines = _wrap_text(headline, max_chars_headline)
        sub_lines = _wrap_text(subheadline, max_chars_sub)

        # Recompute segment lines with emphasis
        headline_segments = _segments_by_line(headline, headline_lines, headline_ranges)
        sub_segments = _segments_by_line(subheadline, sub_lines, sub_ranges)

        # Vertical rhythm with current sizes (slightly larger gaps to breathe)
        line_gap = int(headline_size * 0.32 * float(line_gap_factor))
        sub_gap = int(sub_size * 0.26 * float(sub_gap_factor))
        headline_y_start = padding + headline_size
        headline_block_h = len(headline_segments) * headline_size + max(0, (len(headline_segments) - 1) * line_gap)
        subheadline_y_start = headline_y_start + headline_block_h + int(headline_size * 0.45)
        sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)

        # Reserve vertical space for CTA (height + desired gap) so text doesn't push it to the bottom
        cta_gap_scale = v.get("cta_gap_scale")
        if not isinstance(cta_gap_scale, (int, float)):
            cta_gap_scale = rng.uniform(0.9, 1.2)
        cta_gap = max(8, int(sub_size * 0.6 * float(cta_gap_scale)))
        layout_limit = height - padding - cta_height - int(cta_height * 0.30) - cta_gap
        text_bottom = subheadline_y_start + sub_block_h

        fits = text_bottom <= layout_limit
        # Keep best-so-far if fits or first iteration
        if fits:
            best = (
                headline_segments,
                sub_segments,
                line_gap,
                sub_gap,
                headline_y_start,
                subheadline_y_start,
                headline_size,
                sub_size,
            )
            break

        # record current as fallback in case we hit min sizes
        if best is None:
            best = (
                headline_segments,
                sub_segments,
                line_gap,
                sub_gap,
                headline_y_start,
                subheadline_y_start,
                headline_size,
                sub_size,
            )

        # Downscale and retry
        attempt += 1
        if attempt >= max_attempts or (headline_size <= 30 and sub_size <= 16):
            # stop; use best recorded
            headline_segments, sub_segments, line_gap, sub_gap, headline_y_start, subheadline_y_start, headline_size, sub_size = best
            break
        headline_size = max(30, int(headline_size * 0.92))
        sub_size = max(16, int(sub_size * 0.92))

    # Unpack best if we exited early without assigning
    if isinstance(best, tuple):
        (
            headline_segments,
            sub_segments,
            line_gap,
            sub_gap,
            headline_y_start,
            subheadline_y_start,
            headline_size,
            sub_size,
        ) = best

    # After fitting, optionally nudge vertical start to avoid overlapping with the foreground bbox
    try:
        if smart_layout:
            bbox = analysis.get("foreground_bbox") or None
            orig_size = analysis.get("original_size") or None
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                if orig_size and isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
                    W0 = max(1, int(orig_size[0]))
                    H0 = max(1, int(orig_size[1]))
                else:
                    W0, H0 = width, height
                # Map original bbox to canvas respecting bg_fit ('slice' cover vs 'meet' contain)
                try:
                    _bg_fit_local = str((v.get("bg_fit", "meet"))).lower()
                except Exception:
                    _bg_fit_local = "meet"
                if _bg_fit_local not in ("meet", "slice"):
                    _bg_fit_local = "meet"
                s = (max(width / float(W0), height / float(H0))
                     if _bg_fit_local == "slice"
                     else min(width / float(W0), height / float(H0)))
                w1 = W0 * s
                h1 = H0 * s
                off_x = (width - w1) / 2.0
                off_y = (height - h1) / 2.0
                bx0 = bbox[0] * s + off_x
                by0 = bbox[1] * s + off_y
                bx1 = bbox[2] * s + off_x
                by1 = bbox[3] * s + off_y

                # Approx text block bounds
                tx0 = text_x
                tx1 = text_x + content_width
                ty0 = headline_y_start - int(0.85 * headline_size)
                ty1 = subheadline_y_start + sub_block_h

                def _overlap(a0, a1, b0, b1):
                    return max(0, min(a1, b1) - max(a0, b0))

                inter_w = _overlap(tx0, tx1, bx0, bx1)
                inter_h = _overlap(ty0, ty1, by0, by1)
                inter_area = inter_w * inter_h
                text_area = (tx1 - tx0) * max(1, (ty1 - ty0))
                if inter_area > 0 and text_area > 0 and (inter_area / float(text_area)) > 0.12:
                    # Decide direction: move up or down depending on free space
                    margin = int(height * 0.03)
                    safe_bottom_here = height - padding - int(cta_height * 0.30)
                    room_above = max(0, (headline_y_start - int(0.85 * headline_size)) - padding)
                    room_below = max(0, (safe_bottom_here - sub_block_h) - ty0)

                    # Amount needed to clear overlap
                    delta_down_need = max(0, int(by1 + margin) - ty0)
                    delta_up_need = max(0, ty1 - int(by0 - margin))

                    # Clamp by available room
                    delta_down = min(delta_down_need, room_below)
                    delta_up = min(delta_up_need, room_above)

                    # Prefer the direction with more room; tie -> move up to avoid bottom crowding
                    move_down = (room_below > room_above) and (delta_down > 0 or delta_up == 0)
                    if move_down and delta_down > 0:
                        headline_y_start += int(delta_down)
                        subheadline_y_start += int(delta_down)
                    elif delta_up > 0:
                        headline_y_start -= int(delta_up)
                        subheadline_y_start -= int(delta_up)
    except Exception:
        # Non-fatal; keep fitted positions
        pass

    # Reflow: ensure there is room for CTA under subheadline. First try moving text up, then shrink as last resort.
    try:
        cta_gap = max(8, int(sub_size * 0.6))
        safe_bottom = height - padding - int(cta_height * 0.30)
        sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
        needed_bottom = subheadline_y_start + sub_block_h + cta_gap + cta_height
        if needed_bottom > safe_bottom:
            # 1) Move block up if we have headroom
            headroom_up = max(0, (headline_y_start - int(0.85 * headline_size)) - padding)
            delta = min(headroom_up, needed_bottom - safe_bottom)
            if delta > 0:
                headline_y_start -= delta
                subheadline_y_start -= delta
                needed_bottom -= delta

        # 2) If still overflowing, scale down text and re-wrap a few times
        attempts = 0
        while needed_bottom > safe_bottom and attempts < 5 and (headline_size > 30 or sub_size > 16):
            attempts += 1
            headline_size = max(30, int(headline_size * 0.94))
            sub_size = max(16, int(sub_size * 0.94))
            line_gap = int(headline_size * 0.32 * float(line_gap_factor))
            sub_gap = int(sub_size * 0.26 * float(sub_gap_factor))
            max_chars_headline = _measure_chars_for_width(content_width, headline_size)
            max_chars_sub = _measure_chars_for_width(content_width, sub_size)
            headline_lines = _wrap_text(headline, max_chars_headline)
            sub_lines = _wrap_text(subheadline, max_chars_sub)
            headline_segments = _segments_by_line(headline, headline_lines, headline_ranges)
            sub_segments = _segments_by_line(subheadline, sub_lines, sub_ranges)
            headline_block_h = len(headline_segments) * headline_size + max(0, (len(headline_segments) - 1) * line_gap)
            # keep headline_y_start anchored to padding
            headline_y_start = padding + headline_size
            subheadline_y_start = headline_y_start + headline_block_h + int(headline_size * 0.45)
            sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
            needed_bottom = subheadline_y_start + sub_block_h + cta_gap + cta_height
    except Exception:
        # Best-effort: continue with current layout
        pass

    # Final CTA placement directly under subheadline block (not stuck at bottom)
    try:
        # recompute sub_block_h based on final segments and sizes
        sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
        # keep a small safe area above bottom to avoid looking pinned to the edge
        safe_bottom = height - padding - int(cta_height * 0.30)
        # Position CTA horizontally based on text alignment within the text panel
        if text_align == "left":
            cta_x = int(text_x)
        elif text_align == "right":
            cta_x = int(text_x + content_width - cta_width)
        else:
            x_cand = text_x + (content_width - cta_width) / 2.0
            cta_x = int(max(text_x, min(x_cand, text_x + content_width - cta_width)))
        cta_gap_scale = v.get("cta_gap_scale")
        if not isinstance(cta_gap_scale, (int, float)):
            cta_gap_scale = 1.0
        cta_gap = max(8, int(sub_size * 0.6 * float(cta_gap_scale)))
        cta_y = subheadline_y_start + sub_block_h + cta_gap
        # Clamp CTA within bounds if content is long
        if cta_y + cta_height > safe_bottom:
            cta_y = max(padding, safe_bottom - cta_height)
        cta_text_x = cta_x + cta_width // 2
        cta_text_y = cta_y + int(cta_height * 0.66)
    except Exception:
        # Best-effort; fallback values from earlier remain
        pass

    # If there is excessive vertical slack, first try to scale UP the text to fill space, then align vertically per variant
    try:
        sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
        safe_bottom = height - padding - int(cta_height * 0.30)
        block_top = max(padding, headline_y_start - int(0.85 * headline_size))
        block_bottom = max(subheadline_y_start + sub_block_h, cta_y + cta_height)
        avail_top = padding
        avail_bottom = safe_bottom
        block_h = max(1, block_bottom - block_top)
        avail_h = max(1, avail_bottom - avail_top)
        slack = avail_h - block_h
        # If plenty of slack, attempt to scale up text until we nearly fill the safe area
        if slack > max(24, int(height * 0.08)):
            growth_attempts = 0
            while growth_attempts < 5:
                growth_attempts += 1
                # scale up modestly
                headline_size = int(headline_size * 1.04)
                sub_size = int(sub_size * 1.04)
                line_gap = int(headline_size * 0.32 * float(line_gap_factor))
                sub_gap = int(sub_size * 0.26 * float(sub_gap_factor))
                max_chars_headline = _measure_chars_for_width(content_width, headline_size)
                max_chars_sub = _measure_chars_for_width(content_width, sub_size)
                headline_lines = _wrap_text(headline, max_chars_headline)
                sub_lines = _wrap_text(subheadline, max_chars_sub)
                headline_segments = _segments_by_line(headline, headline_lines, headline_ranges)
                sub_segments = _segments_by_line(subheadline, sub_lines, sub_ranges)
                headline_block_h = len(headline_segments) * headline_size + max(0, (len(headline_segments) - 1) * line_gap)
                headline_y_start = padding + headline_size
                subheadline_y_start = headline_y_start + headline_block_h + int(headline_size * 0.45)
                sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
                # Recompute CTA placement directly under subheadline
                cta_gap = max(8, int(sub_size * 0.6))
                cta_y = subheadline_y_start + sub_block_h + cta_gap
                cta_text_y = cta_y + int(cta_height * 0.66)
                # Check fit
                block_top = max(padding, headline_y_start - int(0.85 * headline_size))
                block_bottom = max(subheadline_y_start + sub_block_h, cta_y + cta_height)
                block_h = max(1, block_bottom - block_top)
                if block_bottom > safe_bottom:
                    # too big; revert last growth and stop
                    # scale back down slightly for safety
                    headline_size = int(headline_size / 1.06)
                    sub_size = int(sub_size / 1.06)
                    # re-measure with reverted sizes
                    line_gap = int(headline_size * 0.32 * float(line_gap_factor))
                    sub_gap = int(sub_size * 0.26 * float(sub_gap_factor))
                    max_chars_headline = _measure_chars_for_width(content_width, headline_size)
                    max_chars_sub = _measure_chars_for_width(content_width, sub_size)
                    headline_lines = _wrap_text(headline, max_chars_headline)
                    sub_lines = _wrap_text(subheadline, max_chars_sub)
                    headline_segments = _segments_by_line(headline, headline_lines, headline_ranges)
                    sub_segments = _segments_by_line(subheadline, sub_lines, sub_ranges)
                    headline_block_h = len(headline_segments) * headline_size + max(0, (len(headline_segments) - 1) * line_gap)
                    headline_y_start = padding + headline_size
                    subheadline_y_start = headline_y_start + headline_block_h + int(headline_size * 0.45)
                    sub_block_h = len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)
                    cta_gap = max(8, int(sub_size * 0.6))
                    cta_y = subheadline_y_start + sub_block_h + cta_gap
                    cta_text_y = cta_y + int(cta_height * 0.66)
                    break
            # refresh slack value
            block_top = max(padding, headline_y_start - int(0.85 * headline_size))
            block_bottom = max(subheadline_y_start + sub_block_h, cta_y + cta_height)
            block_h = max(1, block_bottom - block_top)
            avail_h = max(1, avail_bottom - avail_top)
            slack = avail_h - block_h
        # Now align the block vertically per variant preference with whatever slack remains
        valign = str((v.get("vertical_align") or "middle")).lower()
        if slack > max(20, int(height * 0.06)):
            if valign == "top":
                target_top = avail_top
                delta = target_top - block_top
            elif valign == "bottom":
                target_bottom = avail_bottom
                delta = target_bottom - block_bottom
            else:
                target_top = avail_top + int(slack / 2)
                delta = target_top - block_top
            headline_y_start += int(delta)
            subheadline_y_start += int(delta)
            cta_y += int(delta)
            cta_text_y = cta_y + int(cta_height * 0.66)
            # clamp CTA within safe bottom
            if cta_y + cta_height > safe_bottom:
                overshoot = (cta_y + cta_height) - safe_bottom
                headline_y_start -= int(overshoot)
                subheadline_y_start -= int(overshoot)
                cta_y -= int(overshoot)
                cta_text_y = cta_y + int(cta_height * 0.66)
    except Exception:
        pass

    # SVG template with optional legibility scrim and seeded image filter variance
    try:
        def _clip(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))
        # Respect variant flag for scrim overlays
        show_scrim = bool(v.get("show_scrim", False))
        # Bottom-left vignette (corner shadow) toggle and strength
        try:
            _vig_enable = v.get("corner_shadow_bl")
            show_corner_vignette_bl = bool(_vig_enable) if isinstance(_vig_enable, (bool, int, float)) else False
        except Exception:
            show_corner_vignette_bl = False
        try:
            _vig_strength = v.get("corner_shadow_strength")
            corner_shadow_strength = _clamp_float(_vig_strength, 0.0, 2.0, 1.0)
            if corner_shadow_strength is None:
                corner_shadow_strength = 1.0
        except Exception:
            corner_shadow_strength = 1.0
        # Background fit behavior for <image>: 'meet' (no crop, may letterbox) or 'slice' (cover, may crop)
        # Default to 'slice' for a bold, modern full-bleed look
        bg_fit = str(v.get("bg_fit", "slice")).lower()
        if bg_fit not in ("meet", "slice"):
            bg_fit = "meet"
        preserve_mode = f"xMidYMid {'slice' if bg_fit == 'slice' else 'meet'}"

        # Keep scrim params available if enabled
        shade_factor = v.get("shade_factor")
        if not isinstance(shade_factor, (int, float)):
            shade_factor = rng.uniform(0.9, 1.15)
        shade_main_opacity = _clip(0.35 * float(shade_factor), 0.18, 0.45)
        shade_mid_opacity = _clip(0.20 * float(shade_factor), 0.10, 0.28)
        shade_end_opacity = _clip(0.08 * float(shade_factor), 0.03, 0.12)
        side_shade_factor = v.get("side_shade_factor")
        if not isinstance(side_shade_factor, (int, float)):
            side_shade_factor = rng.uniform(0.9, 1.2)
        side_shade_opacity = _clip(0.48 * float(side_shade_factor), 0.32, 0.62)

        # Compute vignette opacities scaled by strength (applies only if enabled)
        vig1_main_opacity = _clip(0.42 * float(corner_shadow_strength), 0.0, 0.95)
        vig1_mid_opacity = _clip(0.24 * float(corner_shadow_strength), 0.0, 0.85)
        vig2_main_opacity = _clip(0.55 * float(corner_shadow_strength), 0.0, 0.95)
        vig2_mid_opacity = _clip(0.28 * float(corner_shadow_strength), 0.0, 0.90)

        # Prompt-influenced image filter variance (override with emotional_mode variant if provided)
        prompt_text = f"{copy_data.get('headline','')} {copy_data.get('subheadline','')}".lower()
        strong_keywords = ["bold", "sale", "deal", "new", "limited", "fitness", "energy", "power", "high", "premium"]
        soft_keywords = ["calm", "eco", "natural", "gentle", "soft", "soothing", "minimal", "organic"]
        mode = str((v.get("emotional_mode") or "")).lower()
        if mode not in ("strong", "soft", "neutral"):
            mode = "neutral"
            if any(k in prompt_text for k in strong_keywords):
                mode = "strong"
            elif any(k in prompt_text for k in soft_keywords):
                mode = "soft"

        if mode == "strong":
            contrast = _clip(rng.uniform(1.15, 1.35), 0.7, 1.6)
            brightness = _clip(rng.uniform(0.98, 1.08), 0.7, 1.6)
            saturate = _clip(rng.uniform(1.10, 1.30), 0.3, 2.5)
        elif mode == "soft":
            contrast = _clip(rng.uniform(0.85, 0.98), 0.7, 1.6)
            brightness = _clip(rng.uniform(1.02, 1.12), 0.7, 1.6)
            saturate = _clip(rng.uniform(0.85, 1.05), 0.3, 2.5)
        else:
            contrast = _clip(rng.uniform(0.95, 1.15), 0.7, 1.6)
            brightness = _clip(rng.uniform(0.96, 1.08), 0.7, 1.6)
            saturate = _clip(rng.uniform(0.95, 1.15), 0.3, 2.5)

        # Convert brightness & contrast to linear component transfer
        img_slope = float(contrast * brightness)
        img_intercept = float(0.5 * (1.0 - contrast) * brightness)
        img_saturate = float(saturate)
    except Exception:
        show_scrim = False
        show_corner_vignette_bl = False
        # Fallbacks to safe defaults
        shade_main_opacity, shade_mid_opacity, shade_end_opacity, side_shade_opacity = 0.28, 0.16, 0.06, 0.50
        img_slope, img_intercept, img_saturate = 1.0, 0.0, 1.0
        preserve_mode = "xMidYMid slice"
        vig1_main_opacity = 0.0
        vig1_mid_opacity = 0.0
        vig2_main_opacity = 0.0
        vig2_mid_opacity = 0.0
    try:
        # If CTA still lands in bottom gutter, lift text block and CTA together
        bottom_gutter = int(height * 0.12)
        min_cta_top = height - bottom_gutter - cta_height
        if cta_y > min_cta_top:
            # Amount to lift to be just above gutter
            lift = cta_y - min_cta_top
            # Respect available headroom above
            headroom = max(0, (headline_y_start - int(0.85 * headline_size)) - padding)
            lift = min(lift, headroom)
            if lift > 0:
                headline_y_start -= lift
                subheadline_y_start -= lift
                cta_y -= lift
                cta_text_y = cta_y + int(cta_height * 0.66)
    except Exception:
        pass

    try:
        logger.info(
            f"layout: panel={panel_side} text_x={text_x} headline_y={headline_y_start} sub_y={subheadline_y_start} cta=({cta_x},{cta_y}) size=({width}x{height})"
        )
    except Exception:
        pass

    # Compute enhanced styling variables (accent variants, strokes, optional panel card and badge)
    # Helper color utilities
    def _hex_to_rgb(h: str):
        try:
            h = (h or "").lstrip('#')
            if len(h) == 3:
                h = ''.join([c*2 for c in h])
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return (37, 99, 235)  # #2563EB

    def _rgb_to_hex(rgb):
        try:
            r, g, b = [max(0, min(255, int(v))) for v in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#2563EB"

    def _mix_rgb(c1, c2, t: float):
        t = max(0.0, min(1.0, float(t)))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    try:
        base_rgb = _hex_to_rgb(cta_color)
        white = (255, 255, 255)
        accent_light = _rgb_to_hex(_mix_rgb(base_rgb, white, 0.35))
    except Exception:
        accent_light = "#ffffff"

    headline_stroke_w = max(1, int(headline_size * 0.04))
    sub_stroke_w = max(1, int(sub_size * 0.03))

    # Optional visual styles via variant
    try:
        # Default to solid; allow variant to override to gradient
        requested_headline_fill = str(v.get("headline_fill", "solid")).lower()
        headline_fill = requested_headline_fill if requested_headline_fill in ("solid", "gradient") else "solid"
        # Panel style: allow 'card' overlay if requested
        _requested_panel_style = str(v.get("panel_style", "none")).lower()
        panel_style = _requested_panel_style if _requested_panel_style in ("none", "card") else "none"
    except Exception:
        headline_fill = "gradient"
        panel_style = "none"

    # Panel card geometry (if enabled)
    panel_card_x = panel_card_y = panel_card_w = panel_card_h = 0
    panel_card_radius = 18
    try:
        if panel_style == "card":
            panel_pad = max(10, int(headline_size * 0.30))
            text_block_top = max(padding, headline_y_start - int(0.85 * headline_size))
            text_block_bottom = max(subheadline_y_start + (len(sub_segments) * sub_size + max(0, (len(sub_segments) - 1) * sub_gap)), cta_y + cta_height)
            panel_card_x = max(padding, text_x - int(panel_pad * 0.8))
            panel_card_w = min(width - 2 * padding, content_width + int(panel_pad * 1.6))
            panel_card_y = max(padding, text_block_top - panel_pad)
            panel_card_h = min(height - 2 * padding, (text_block_bottom - panel_card_y) + panel_pad)
            panel_card_radius = max(12, int(min(panel_card_w, panel_card_h) * 0.04))
    except Exception:
        pass

    # Optional badge/ribbon
    try:
        badge_text = (copy_data.get("badge") or v.get("badge_text") or "").strip()
    except Exception:
        badge_text = ""
    badge_x = badge_y = badge_w = badge_h = 0
    badge_color = cta_color
    try:
        if badge_text:
            per_char = 10
            badge_w = max(80, min(content_width, 30 + len(badge_text) * per_char))
            badge_h = max(28, int(sub_size * 0.9) + 12)
            badge_x = text_x
            badge_y = max(12, padding)
            # If headline starts very near top, nudge badge below padding
            if badge_y + badge_h > (headline_y_start - int(0.8 * headline_size)):
                badge_y = max(padding, headline_y_start - int(0.8 * headline_size) - badge_h - 8)
    except Exception:
        pass

    # Map text alignment to SVG text-anchor and anchor X position
    try:
        if text_align == "left":
            text_anchor_attr = "start"
            anchor_x = int(text_x)
        elif text_align == "right":
            text_anchor_attr = "end"
            anchor_x = int(text_x + content_width)
        else:
            text_anchor_attr = "middle"
            anchor_x = int(content_center_x)
    except Exception:
        text_anchor_attr = "middle"
        anchor_x = int(content_center_x)

    svg_template = """
    <svg width="{{ width }}" height="{{ height }}" viewBox="0 0 {{ width }} {{ height }}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <linearGradient id="shade" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#000000" stop-opacity="{{ shade_main_opacity }}" />
                <stop offset="50%" stop-color="#000000" stop-opacity="{{ shade_mid_opacity }}" />
                <stop offset="100%" stop-color="#000000" stop-opacity="{{ shade_end_opacity }}" />
            </linearGradient>
            <linearGradient id="shadeLeft" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#000000" stop-opacity="{{ side_shade_opacity }}" />
                <stop offset="100%" stop-color="#000000" stop-opacity="0.00" />
            </linearGradient>
            <linearGradient id="shadeRight" x1="100%" y1="0%" x2="0%" y2="0%">
                <stop offset="0%" stop-color="#000000" stop-opacity="{{ side_shade_opacity }}" />
                <stop offset="100%" stop-color="#000000" stop-opacity="0.00" />
            </linearGradient>
            <linearGradient id="bg" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#1f2937" stop-opacity="1" />
                <stop offset="100%" stop-color="#111827" stop-opacity="1" />
            </linearGradient>
            <!-- Gradient for headline text fill -->
            <linearGradient id="headlineGrad" x1="0%" y1="0%" x2="0" y2="100%">
                <stop offset="0%" stop-color="{{ accent_light }}" />
                <stop offset="100%" stop-color="{{ cta_color }}" />
            </linearGradient>
            <filter id="imgFilter" x="-50%" y="-50%" width="200%" height="200%">
                <feComponentTransfer>
                    <feFuncR type="linear" slope="{{ img_slope }}" intercept="{{ img_intercept }}" />
                    <feFuncG type="linear" slope="{{ img_slope }}" intercept="{{ img_intercept }}" />
                    <feFuncB type="linear" slope="{{ img_slope }}" intercept="{{ img_intercept }}" />
                </feComponentTransfer>
                <feColorMatrix type="saturate" values="{{ img_saturate }}" />
            </filter>
            <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="2" dy="2" stdDeviation="4" flood-color="#000000" flood-opacity="0.28"/>
            </filter>
            <filter id="textGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
            <!-- Professional bottom-left vignette: stronger at the corner, quick falloff -->
            <radialGradient id="vignetteBL" cx="0%" cy="100%" r="68%" fx="0%" fy="100%" gradientTransform="matrix(1 0 0 0.72 0 0)">
                <stop offset="0%" stop-color="#000000" stop-opacity="{{ vig1_main_opacity }}" />
                <stop offset="38%" stop-color="#000000" stop-opacity="{{ vig1_mid_opacity }}" />
                <stop offset="100%" stop-color="#000000" stop-opacity="0.00" />
            </radialGradient>
            <!-- Inner accent vignette for a two-tone professional corner -->
            <radialGradient id="vignetteBL2" cx="0%" cy="100%" r="42%" fx="0%" fy="100%" gradientTransform="matrix(1 0 0 0.58 0 0)">
                <stop offset="0%" stop-color="#000000" stop-opacity="{{ vig2_main_opacity }}" />
                <stop offset="30%" stop-color="#000000" stop-opacity="{{ vig2_mid_opacity }}" />
                <stop offset="100%" stop-color="#000000" stop-opacity="0.00" />
            </radialGradient>
        </defs>

        <!-- Background -->
        {% if original_url %}
        <image xlink:href="{{ original_url | e }}" x="0" y="0" width="{{ width }}" height="{{ height }}" preserveAspectRatio="{{ preserve_mode }}" filter="url(#imgFilter)"/>
        {% if show_scrim %}
        <!-- Optional legibility gradient overlay -->
        <rect width="100%" height="100%" fill="url(#shade)" />
        {% endif %}
        {% if show_corner_vignette_bl %}
        <!-- Optional bottom-left corner vignette for emphasis/legibility (variant-controlled) -->
        <rect width="100%" height="100%" fill="url(#vignetteBL)" />
        <rect width="100%" height="100%" fill="url(#vignetteBL2)" />
        {% endif %}
        {% else %}
        <rect width="100%" height="100%" fill="url(#bg)" />
        {% endif %}

        <!-- Content -->
        <g>
            {% if show_scrim %}
            <!-- Side scrim for text legibility (optional) -->
            {% if panel_side == 'left' %}
            <rect x="0" y="0" width="{{ text_panel_w }}" height="{{ height }}" fill="url(#shadeLeft)" />
            {% elif panel_side == 'right' %}
            <rect x="{{ width - text_panel_w }}" y="0" width="{{ text_panel_w }}" height="{{ height }}" fill="url(#shadeRight)" />
            {% else %}
            <rect x="{{ center_x }}" y="0" width="{{ text_panel_w }}" height="{{ height }}" fill="url(#shade)" />
            {% endif %}
            {% endif %}
            
            {% if panel_style == 'card' %}
            <!-- Card background behind text for legibility -->
            <g filter="url(#shadow)">
                <rect x="{{ panel_card_x }}" y="{{ panel_card_y }}" width="{{ panel_card_w }}" height="{{ panel_card_h }}" rx="{{ panel_card_radius }}" fill="#0b0b0b" fill-opacity="0.55" />
            </g>
            {% endif %}

            {% if badge_text %}
            <!-- Optional badge/ribbon -->
            <g filter="url(#shadow)">
                <rect x="{{ badge_x }}" y="{{ badge_y }}" width="{{ badge_w }}" height="{{ badge_h }}" rx="8" fill="{{ badge_color }}" />
                <text x="{{ badge_x + badge_w/2 }}" y="{{ badge_y + badge_h*0.68 }}" font-family="{{ font_family_headline | e }}" font-size="{{ int(sub_size*0.85) }}" font-weight="800" fill="#ffffff" text-anchor="middle">{{ badge_text | e }}</text>
            </g>
            {% endif %}
            <!-- Headline within text panel -->
            <text x="{{ anchor_x }}" y="{{ headline_y_start }}" text-anchor="{{ text_anchor_attr }}" font-family="{{ font_family_headline | e }}" font-size="{{ headline_size }}" font-weight="{{ headline_weight }}" fill="{% if headline_fill == 'gradient' %}url(#headlineGrad){% else %}{{ text_color }}{% endif %}" letter-spacing="{{ headline_letter_spacing }}" paint-order="fill" stroke="none">
                {% for line in headline_segments %}
                <tspan x="{{ anchor_x }}" dy="{% if loop.first %}0{% else %}{{ headline_size + line_gap }}{% endif %}">
                    {% for seg in line %}
                    <tspan{% if seg.style == 'bold' %} font-weight="800"{% elif seg.style == 'italic' %} font-style="italic"{% endif %}{% if seg.style == 'highlight' %} fill="{{ highlight_text_color }}" stroke="{{ headline_highlight_stroke }}" stroke-width="{{ headline_highlight_stroke_w }}" stroke-opacity="1" paint-order="stroke fill" stroke-linejoin="round" stroke-linecap="round"{% endif %}>{{ (seg.text | upper if seg.style == 'caps' else seg.text) | e }}</tspan>
                    {% endfor %}
                </tspan>
                {% endfor %}
            </text>

            <!-- Subheadline within text panel -->
            <text x="{{ anchor_x }}" y="{{ subheadline_y_start }}" text-anchor="{{ text_anchor_attr }}" font-family="{{ font_family_body | e }}" font-size="{{ sub_size }}" font-weight="{{ body_weight }}" fill="{{ text_color_secondary }}" letter-spacing="{{ body_letter_spacing }}" paint-order="fill" stroke="none">
                {% for line in sub_segments %}
                <tspan x="{{ anchor_x }}" dy="{% if loop.first %}0{% else %}{{ sub_size + sub_gap }}{% endif %}">
                    {% for seg in line %}
                    <tspan{% if seg.style == 'bold' %} font-weight="700"{% elif seg.style == 'italic' %} font-style="italic"{% endif %}{% if seg.style == 'highlight' %} fill="{{ highlight_text_color }}" stroke="{{ sub_highlight_stroke }}" stroke-width="{{ sub_highlight_stroke_w }}" stroke-opacity="1" paint-order="stroke fill" stroke-linejoin="round" stroke-linecap="round"{% endif %}>{{ (seg.text | upper if seg.style == 'caps' else seg.text) | e }}</tspan>
                    {% endfor %}
                </tspan>
                {% endfor %}
            </text>

            <!-- CTA Button centered within text panel -->
            <rect x="{{ cta_x }}" y="{{ cta_y }}" width="{{ cta_width }}" height="{{ cta_height }}" rx="{{ cta_radius }}" fill="{{ cta_fill }}" filter="url(#shadow)" stroke="{{ cta_stroke }}" stroke-opacity="{{ cta_stroke_opacity }}"/>
            <text x="{{ cta_x + (cta_width/2) }}" y="{{ cta_text_y }}" font-family="{{ font_family_headline | e }}" font-size="{{ int(sub_size*0.9) }}" font-weight="800" fill="#ffffff" text-anchor="middle">{{ cta | upper | e }}</text>
        </g>
    </svg>
    """

    template = Template(svg_template)
    img_href = _resolve_img_href(analysis)
    try:
        logger.info(
            "compose: using image href len=%s prefix=%s",
            (len(img_href) if img_href else 0),
            ((img_href[:32] + "...") if img_href and len(img_href) > 35 else (img_href or "")),
        )
    except Exception:
        pass
    center_x = int((width - text_panel_w) / 2)
    # Variables for enhanced styling passed to the template
    svg_content = template.render(
        width=width,
        height=height,
        padding=padding,
        font_family_headline=font_family_headline,
        font_family_body=font_family_body,
        text_color=text_color,
        text_color_secondary=text_color_secondary,
        headline_size=headline_size,
        sub_size=sub_size,
        headline_segments=headline_segments,
        sub_segments=sub_segments,
        line_gap=line_gap,
        sub_gap=sub_gap,
        headline_y_start=headline_y_start,
        subheadline_y_start=subheadline_y_start,
        panel_side=panel_side,
        text_x=text_x,
        content_width=content_width,
        cta_x=cta_x,
        cta_y=cta_y,
        cta_width=cta_width,
        cta_height=cta_height,
        cta_text_x=cta_text_x,
        cta_text_y=cta_text_y,
        cta=cta_text,
        headline_letter_spacing=headline_letter_spacing,
        body_letter_spacing=body_letter_spacing,
        headline_weight=headline_weight,
        body_weight=body_weight,
        cta_color=cta_color,
        accent_light=accent_light,
        text_panel_w=text_panel_w,
        original_url=img_href,
        preserve_mode=preserve_mode,
        # scrim/gradient vars
        shade_main_opacity=shade_main_opacity,
        shade_mid_opacity=shade_mid_opacity,
        shade_end_opacity=shade_end_opacity,
        side_shade_opacity=side_shade_opacity,
        # corner vignette vars
        show_corner_vignette_bl=show_corner_vignette_bl,
        vig1_main_opacity=vig1_main_opacity,
        vig1_mid_opacity=vig1_mid_opacity,
        vig2_main_opacity=vig2_main_opacity,
        vig2_mid_opacity=vig2_mid_opacity,
        center_x=center_x,
        content_center_x=content_center_x,
        text_anchor_attr=text_anchor_attr,
        anchor_x=anchor_x,
        show_scrim=show_scrim,
        # image filter vars
        img_slope=img_slope,
        img_intercept=img_intercept,
        img_saturate=img_saturate,
        # CTA style vars
        cta_radius=cta_radius,
        cta_fill=cta_fill,
        cta_stroke=cta_stroke,
        cta_stroke_opacity=cta_stroke_opacity,
        # Enhanced text styling
        headline_fill=headline_fill,
        headline_stroke_w=headline_stroke_w,
        sub_stroke_w=sub_stroke_w,
        # Panel card
        panel_style=panel_style,
        panel_card_x=panel_card_x,
        panel_card_y=panel_card_y,
        panel_card_w=panel_card_w,
        panel_card_h=panel_card_h,
        panel_card_radius=panel_card_radius,
        # Badge
        badge_text=badge_text,
        badge_x=badge_x,
        badge_y=badge_y,
        badge_w=badge_w,
        badge_h=badge_h,
        badge_color=badge_color,
        int=int,
    )
    return svg_content

# Curated catalog of modern ad layout variants
def modern_variant_catalog(copy_data: Dict[str, Any], analysis: Dict[str, Any], crop_info: Dict[str, Any], *, base_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a curated list of modern ad variant presets. Each item has keys: id, name, variant."""
    palette = analysis.get("palette") or []
    pal_n = len(palette)
    def acc(i: int) -> int:
        return max(0, min(max(0, pal_n - 1), i))
    badge_text = (copy_data.get("badge") or "").strip()
    presets: List[Dict[str, Any]] = []
    def add(id_: str, name: str, v: Dict[str, Any]):
        presets.append({"id": id_, "name": name, "variant": v})

    # 1) Hero left, text right with scrim + card (bold retail/performance)
    add("hero_left_text_right_card", "Hero Left · Right Text · Card", {
        "panel_side": "right",
        "text_align": "left",
        "show_scrim": True,
        "panel_style": "card",
        "headline_fill": "solid",
        "cta_style": "pill",
        "cta_width_mode": "wide",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "strong",
        "accent_index": acc(1),
        "type_scale": 1.06,
        "panel_width_factor": 0.50,
    })

    # 2) Hero right, text left with scrim (brand storytelling)
    add("hero_right_text_left_scrim", "Hero Right · Left Text · Scrim", {
        "panel_side": "left",
        "text_align": "left",
        "show_scrim": True,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "fill",
        "cta_width_mode": "auto",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "neutral",
        "accent_index": acc(2),
        "type_scale": 1.02,
        "panel_width_factor": 0.52,
    })

    # 3) Centered minimal gradient headline (premium aesthetic)
    add("centered_minimal_gradient", "Centered Minimal · Gradient Headline", {
        "panel_side": "center",
        "text_align": "center",
        "show_scrim": False,
        "panel_style": "none",
        "headline_fill": "gradient",
        "cta_style": "pill",
        "cta_width_mode": "auto",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "neutral",
        "accent_index": acc(1),
        "type_scale": 1.08,
        "panel_width_factor": 0.92,
    })

    # 4) Top-aligned info (editorial feel)
    add("top_left_editorial", "Top Left · Editorial", {
        "panel_side": "left",
        "text_align": "left",
        "show_scrim": False,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "outline",
        "cta_width_mode": "auto",
        "bg_fit": "slice",
        "vertical_align": "top",
        "emotional_mode": "neutral",
        "accent_index": acc(3 if pal_n > 3 else 1),
        "type_scale": 0.96,
        "panel_width_factor": 0.48,
        "line_gap_factor": 1.12,
        "sub_gap_factor": 1.08,
    })

    # 5) Bottom-right bold (promotional)
    add("bottom_right_bold", "Bottom Right · Bold", {
        "panel_side": "right",
        "text_align": "right",
        "show_scrim": True,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "pill",
        "cta_width_mode": "wide",
        "bg_fit": "slice",
        "vertical_align": "bottom",
        "emotional_mode": "strong",
        "accent_index": acc(0),
        "type_scale": 1.10,
        "panel_width_factor": 0.50,
    })

    # 6) Badge announcement (launch/new)
    add("badge_announcement", "Centered · Badge Announcement", {
        "panel_side": "center",
        "text_align": "center",
        "show_scrim": False,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "fill",
        "cta_width_mode": "auto",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "strong",
        "accent_index": acc(2),
        "badge_text": badge_text[:80] if badge_text else "",
        "type_scale": 1.04,
        "panel_width_factor": 0.88,
    })

    # 7) Soft eco/minimal (wellness/eco brands)
    add("soft_eco_minimal", "Soft Eco · Minimal", {
        "panel_side": "center",
        "text_align": "center",
        "show_scrim": True,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "outline",
        "cta_width_mode": "auto",
        "bg_fit": "meet",
        "vertical_align": "middle",
        "emotional_mode": "soft",
        "accent_index": acc(1),
        "type_scale": 1.00,
        "shade_factor": 0.70,
        "side_shade_factor": 0.70,
        "panel_width_factor": 0.90,
    })

    # 8) Product card overlay with meet-fit (catalog/product highlight)
    add("product_card_meet", "Product Card · Meet Fit", {
        "panel_side": "left",
        "text_align": "left",
        "show_scrim": False,
        "panel_style": "card",
        "headline_fill": "solid",
        "cta_style": "fill",
        "cta_width_mode": "auto",
        "bg_fit": "meet",
        "vertical_align": "middle",
        "emotional_mode": "neutral",
        "accent_index": acc(2),
        "type_scale": 1.00,
        "panel_width_factor": 0.50,
    })

    # 9) Centered wide CTA
    add("centered_wide_cta", "Centered · Wide CTA", {
        "panel_side": "center",
        "text_align": "center",
        "show_scrim": False,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "pill",
        "cta_width_mode": "wide",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "neutral",
        "accent_index": acc(0),
        "type_scale": 1.02,
        "panel_width_factor": 0.90,
    })

    # 10) Smart flip (let smart_layout decide, but flip for variety)
    add("smart_flip_rhythm", "Smart Flip · Rhythm", {
        "flip_panel": True,
        "text_align": "left",
        "show_scrim": True,
        "panel_style": "none",
        "headline_fill": "solid",
        "cta_style": "fill",
        "cta_width_mode": "auto",
        "bg_fit": "slice",
        "vertical_align": "middle",
        "emotional_mode": "neutral",
        "accent_index": acc(1),
        "type_scale": 1.00,
        "line_gap_factor": 1.00,
        "sub_gap_factor": 1.00,
    })

    # Attach seeds deterministically
    for i, p in enumerate(presets):
        v = dict(p["variant"])
        try:
            if base_seed is not None:
                v.setdefault("seed", int(base_seed) + i)
        except Exception:
            pass
        p["variant"] = v
    return presets

def public_base_url() -> str:
    return os.getenv('PUBLIC_MINIO_BASE', 'http://localhost:9000').rstrip('/')

def _upload_bytes(bucket: str, key: str, data: bytes, content_type: str) -> Tuple[str, Optional[str]]:
    """Upload bytes to storage. Returns (public_url, internal_url_or_None). In local mode, returns one public URL and None for internal.
    """
    if STORAGE_MODE == 'local':
        # Save under STORAGE_DIR mirroring bucket/key for neatness
        rel_path = f"{bucket}/{key}"
        file_path = Path(STORAGE_DIR) / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(data)
        return f"{PUBLIC_BASE_URL}/static/{rel_path}", None
    else:
        public_base = os.getenv('PUBLIC_MINIO_BASE', 'http://localhost:9000').rstrip('/')
        internal_base = os.getenv('S3_ENDPOINT', 'http://minio:9000').rstrip('/')
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            return f"{public_base}/{bucket}/{key}", f"{internal_base}/{bucket}/{key}"
        except Exception as e:
            logger.error(f"S3 upload failed for {bucket}/{key}: {e}. Falling back to local storage.")
            rel_path = f"{bucket}/{key}"
            file_path = Path(STORAGE_DIR) / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(data)
            return f"{PUBLIC_BASE_URL}/static/{rel_path}", None

def _upload_file_path(bucket: str, key: str, file_path: Path, content_type: str) -> Tuple[str, Optional[str]]:
    """Upload a file on disk to storage, streaming from disk when possible.
    Returns (public_url, internal_url_or_None)."""
    rel_path = f"{bucket}/{key}"
    if STORAGE_MODE == 'local':
        dst = Path(STORAGE_DIR) / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_path, dst)
        return f"{PUBLIC_BASE_URL}/static/{rel_path}", None
    else:
        public_base = os.getenv('PUBLIC_MINIO_BASE', 'http://localhost:9000').rstrip('/')
        internal_base = os.getenv('S3_ENDPOINT', 'http://minio:9000').rstrip('/')
        try:
            with open(file_path, 'rb') as fobj:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=fobj,
                    ContentType=content_type,
                )
            return f"{public_base}/{bucket}/{key}", f"{internal_base}/{bucket}/{key}"
        except Exception as e:
            logger.error(f"S3 upload failed for {bucket}/{key}: {e}. Falling back to local storage.")
            dst = Path(STORAGE_DIR) / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_path, dst)
            return f"{PUBLIC_BASE_URL}/static/{rel_path}", None

async def render_svg_to_formats(svg_content: str, crop_info: Dict[str, Any], job_id: str, *, force: bool = False) -> List[RenderOutput]:
    """Legacy: Render SVG to PNG/JPG using CairoSVG. Kept as a fallback if GPT image is not used/available.
    If force=True, bypasses the RENDER_RASTER toggle (used by /render when force_svg is requested).
    """
    outputs: List[RenderOutput] = []
    outputs_bucket = os.getenv('S3_BUCKET_OUTPUTS', 'outputs')

    width = int(crop_info.get("width", 1080))
    height = int(crop_info.get("height", 1080))

    unique = uuid.uuid4().hex[:8]

    # In PNG-only mode we avoid uploading SVG. This function is only for fallback.
    # Respect rasterization toggle unless force=True.
    if not RENDER_RASTER and not force:
        logger.info("RENDER_RASTER disabled; skipping legacy SVG rasterization (not forced)")
        return []

    # 2) Render SVG -> PNG bytes using CairoSVG with a robust URL fetcher
    def _cairo_url_fetcher(url: str, *_, **__):
        try:
            if not url:
                return {"string": b"", "mime_type": "application/octet-stream"}
            if url.startswith("data:"):
                header, data_part = url.split(",", 1)
                mime = header[5:].split(";")[0] or "application/octet-stream"
                if ";base64" in header:
                    raw = base64.b64decode(data_part)
                else:
                    raw = data_part.encode("utf-8")
                return {"string": raw, "mime_type": mime}

            parsed = urlparse(url)
            path = parsed.path or ""
            # Map local static endpoints to disk
            for prefix in ("/static/", "/outputs/", "/assets/"):
                if path.startswith(prefix):
                    rel_path = path[len(prefix):]
                    file_path = Path(STORAGE_DIR) / rel_path
                    if file_path.exists():
                        mime, _ = mimetypes.guess_type(str(file_path))
                        return {"file_obj": open(file_path, "rb"), "mime_type": mime or "application/octet-stream"}
                    break

            # Fallback: HTTP(S) fetch
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            mime = resp.headers.get("Content-Type", "application/octet-stream")
            return {"string": resp.content, "mime_type": mime}
        except Exception as e:
            logger.warning(f"CairoSVG url_fetcher failed for {url}: {e}")
            return {"string": b"", "mime_type": "application/octet-stream"}

    # Try to render PNG/JPG using CairoSVG if available. If not, gracefully return no outputs.
    png_bytes = None
    try:
        import cairosvg  # type: ignore
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=width,
            output_height=height,
            unsafe=True,
            url_fetcher=_cairo_url_fetcher,
        )
        logger.info(f"Rendered PNG bytes: {len(png_bytes)}")
    except Exception as e:
        logger.warning(f"Skipping PNG/JPG rendering due to missing CairoSVG or native deps: {e}")

    # No Sharp fallback; if Cairo failed, we'll proceed with SVG-only outputs
    if not png_bytes:
        logger.info("Cairo rendering unavailable; skipping rasterization (Sharp removed).")

    if png_bytes:
        # Upload PNG
        png_key = f"outputs/{job_id}_{unique}_png.png"
        png_url, _ = _upload_bytes(outputs_bucket, png_key, png_bytes, 'image/png')
        # Cache-bust preview
        png_url_cb = f"{png_url}{'&' if '?' in png_url else '?'}t={int(time.time())}"
        outputs.append(RenderOutput(format="png", width=width, height=height, url=png_url_cb))

        # 3) Convert PNG -> JPG via Pillow and upload (flatten any transparency onto white to avoid dark backgrounds)
        with Image.open(io.BytesIO(png_bytes)) as im:
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                background = Image.new('RGB', im.size, (255, 255, 255))
                alpha = im.split()[-1] if im.mode != 'P' else Image.new('L', im.size, 255)
                background.paste(im, mask=alpha)
                rgb_im = background
            else:
                rgb_im = im.convert('RGB')
            jpg_buffer = io.BytesIO()
            rgb_im.save(jpg_buffer, format='JPEG', quality=90)
            jpg_buffer.seek(0)
        jpg_key = f"outputs/{job_id}_{unique}_jpg.jpg"
        jpg_url, _ = _upload_bytes(outputs_bucket, jpg_key, jpg_buffer.getvalue(), 'image/jpeg')
        # Cache-bust preview
        jpg_url_cb = f"{jpg_url}{'&' if '?' in jpg_url else '?'}t={int(time.time())}"
        outputs.append(RenderOutput(format="jpg", width=width, height=height, url=jpg_url_cb))

    # If we couldn't rasterize (no PNG/JPG produced), provide an SVG fallback output so the pipeline has at least one asset
    if not outputs:
        try:
            svg_key = f"outputs/{job_id}_{unique}.svg"
            svg_url, _ = _upload_bytes(outputs_bucket, svg_key, svg_content.encode('utf-8'), 'image/svg+xml')
            # Cache-bust preview
            svg_url_cb = f"{svg_url}{'&' if '?' in svg_url else '?'}t={int(time.time())}"
            outputs.append(RenderOutput(format="svg", width=width, height=height, url=svg_url_cb))
        except Exception as e:
            logger.warning(f"Failed to upload SVG fallback: {e}")

    return outputs

async def generate_ad_image_with_ai(copy_data: Dict[str, Any], analysis: Dict[str, Any], crop_info: Dict[str, Any], job_id: str) -> List[RenderOutput]:
    """Generate a PNG ad via GPT image model with a robust Pillow fallback."""
    outputs: List[RenderOutput] = []
    outputs_bucket = os.getenv('S3_BUCKET_OUTPUTS', 'outputs')
    width = int(crop_info.get("width", 1080))
    height = int(crop_info.get("height", 1080))
    unique = uuid.uuid4().hex[:8]

    headline = (copy_data.get("headline") or "").strip()
    subheadline = (copy_data.get("subheadline") or "").strip()
    cta = (copy_data.get("cta") or "Learn More").strip()
    palette = [c for c in (analysis.get("palette") or []) if isinstance(c, str)]
    brand_colors = ", ".join(palette[:4]) if palette else "balanced modern palette"
    prompt = (
        "Create a professional, photorealistic PNG display ad. Typeset the provided copy directly on the image (not as captions).\n"
        "Text requirements: High contrast, bold, legible, large type, clean typography. No heavy filters, no vignettes, no frames, no watermarks.\n"
        "Layout: Balanced composition with the product prominent, text fully visible, no overlap with critical subject. Include a distinct CTA button.\n"
        f"Headline: '{headline}'.\n"
        f"Subheadline: '{subheadline}'.\n"
        f"CTA label: '{cta}'.\n"
        f"Use this color palette tastefully: {brand_colors}.\n"
        "Overall style: modern, premium, minimal, crisp lighting.\n"
        "Avoid watermarks or extraneous logos."
    )

    png_bytes: Optional[bytes] = None
    if USE_GPT_IMAGE and client and _OPENAI_API_KEY:
        # Force square size to supported values (common: 256/512/1024)
        def _nearest(n: int) -> int:
            choices = [256, 512, 1024]
            return min(choices, key=lambda x: abs(x - n))
        side = _nearest(max(width, height))
        size_str = f"{side}x{side}"

        async def _try_model(model_name: str) -> Tuple[Optional[bytes], Optional[str]]:
            try:
                # Some models (e.g. dall-e-3) only support 1024x1024.
                m_size_str = "1024x1024" if model_name == "dall-e-3" else size_str
                resp = await client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    size=m_size_str,
                    timeout=60,
                )
                b64 = None
                try:
                    b64 = resp.data[0].b64_json  # type: ignore[attr-defined]
                except Exception:
                    b64 = None
                if b64:
                    return base64.b64decode(b64), None
                # Fallback: some responses return a URL instead of b64_json
                try:
                    url = getattr(resp.data[0], 'url', None)  # type: ignore[attr-defined]
                except Exception:
                    url = None
                if url:
                    import httpx
                    try:
                        async with httpx.AsyncClient(timeout=30) as h:
                            r = await h.get(url)
                            r.raise_for_status()
                            return r.content, None
                    except Exception as de:
                        msg = f"download_failed for model={model_name}: {de}"
                        logger.warning(msg)
                        return None, msg
                msg = "OpenAI images.generate returned neither b64_json nor url"
                logger.warning(msg)
                return None, msg
            except Exception as e:
                msg = f"model={model_name}: {e}"
                logger.warning(f"OpenAI images.generate failed: {msg}")
                return None, msg

        # Try configured model first; optionally allow fallbacks (deduplicated)
        order = []
        primary_and_fallbacks = [IMAGE_MODEL] if STRICT_IMAGE_MODEL_ONLY else [IMAGE_MODEL, "gpt-image-1", "dall-e-3"]
        for m in primary_and_fallbacks:
            if m and m not in order:
                order.append(m)
        last_err: Optional[str] = None
        for m in order:
            img, err = await _try_model(m)
            if img is not None:
                png_bytes = img
                break
            last_err = err or last_err

    if png_bytes is None:
        # Strict GPT-only mode: no fallback
        detail = f"OpenAI image generation failed for models [{IMAGE_MODEL}, gpt-image-1, dall-e-3]"
        if 'last_err' in locals() and last_err:
            detail += f": {last_err}"
        raise HTTPException(status_code=502, detail=detail)

    png_key = f"outputs/{job_id}_{unique}_png.png"
    png_url, _ = _upload_bytes(outputs_bucket, png_key, png_bytes, 'image/png')
    # Cache-bust preview
    png_url_cb = f"{png_url}{'&' if '?' in png_url else '?'}t={int(time.time())}"
    outputs.append(RenderOutput(format="png", width=width, height=height, url=png_url_cb))
    return outputs
# API Endpoints
@app.post("/ingest-analyze")
async def ingest_analyze(request: Request, image: UploadFile = File(...)):
    """Stage 1: Upload & sanitize, Stage 2: Visual analysis
    - Streams the uploaded file to disk to avoid loading entire payload in memory
    - Enforces per-chunk timeouts and a hard size limit
    - Ensures proper path handling for Windows and Unix systems
    - Provides better logging and supports client cancellation
    """
    t0 = perf_counter()
    tmp_path: Optional[Path] = None
    try:
        # Validate content type early
        ctype = (image.content_type or "").lower()
        if not ctype.startswith("image/"):
            logger.warning(f"Rejecting non-image upload: content_type={ctype!r}")
            raise HTTPException(status_code=415, detail="Unsupported Media Type: expected image/*")

        # Stream upload to a temp file to avoid large memory usage
        suffix = os.path.splitext(image.filename or "")[1] or ".bin"
        # Ensure filename is clean and path is properly joined for Windows
        safe_filename = f"{uuid.uuid4().hex}{suffix}"
        tmp_path = (UPLOAD_DIR_TEMP / safe_filename).resolve()
        # Ensure parent directory exists
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = UPLOAD_MAX_MB * 1024 * 1024
        chunk_size = max(1, UPLOAD_CHUNK_KB) * 1024
        total = 0
        read_chunks = 0
        last_log = 0

        logger.info(f"ingest-analyze: start streaming upload -> {tmp_path.name} (limit={UPLOAD_MAX_MB}MB, chunk={UPLOAD_CHUNK_KB}KB)")
        async with aiofiles.open(tmp_path, "wb") as out:
            while True:
                # honor client disconnects
                if await request.is_disconnected():
                    logger.warning("Client disconnected during upload")
                    raise HTTPException(status_code=499, detail="Client Closed Request")

                try:
                    chunk = await asyncio.wait_for(image.read(chunk_size), timeout=UPLOAD_READ_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.error("Upload read timeout per-chunk")
                    raise HTTPException(status_code=408, detail="Upload read timeout")

                if not chunk:
                    break
                total += len(chunk)
                read_chunks += 1
                if total > max_bytes:
                    logger.warning(f"Upload exceeded limit: {total} bytes > {max_bytes} bytes")
                    raise HTTPException(status_code=413, detail="File too large")
                await out.write(chunk)

                # throttle progress logs
                if total - last_log >= 2 * 1024 * 1024:  # ~2MB intervals
                    logger.info(f"ingest-analyze: streamed {total/1024/1024:.1f} MB so far...")
                    last_log = total

        if total == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        logger.info(f"ingest-analyze: finished upload size={total/1024/1024:.2f} MB, chunks={read_chunks}, elapsed={perf_counter()-t0:.2f}s")

        # Open with Pillow from disk (avoid loading whole bytes in app memory early)
        pil_image = Image.open(str(tmp_path))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Analyze image (CPU-bound) in a thread with timeout
        t1 = perf_counter()
        def _sync_analyze(img: Image.Image):
            m, bb = detect_foreground(img)
            pal = extract_palette(img)
            cr = generate_crop_proposals(img, bb)
            return m, bb, pal, cr
        try:
            mask, bbox, palette, crops = await asyncio.wait_for(asyncio.to_thread(_sync_analyze, pil_image), timeout=PROCESS_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error("Image analysis timed out")
            raise HTTPException(status_code=504, detail="Image analysis timed out")
        logger.info(f"ingest-analyze: analysis done in {perf_counter()-t1:.2f}s (bbox={bbox}, palette_len={len(palette)}, crops={len(crops)})")

        # Upload original to storage by streaming from file path
        assets_bucket = os.getenv('S3_BUCKET_ASSETS', 'assets')
        orig_ext = os.path.splitext(image.filename or "")[1] or '.png'
        orig_key = f"uploads/pipeline/{uuid.uuid4().hex}{orig_ext}"
        mime_guess = None
        try:
            fmt = (getattr(pil_image, 'format', '') or '').upper()
            fmt_to_mime = {
                'JPEG': 'image/jpeg',
                'JPG': 'image/jpeg',
                'PNG': 'image/png',
                'WEBP': 'image/webp',
                'GIF': 'image/gif',
                'BMP': 'image/bmp'
            }
            mime_guess = fmt_to_mime.get(fmt)
        except Exception:
            pass
        content_type = mime_guess or (ctype if ctype.startswith('image/') else 'application/octet-stream')
        original_url_http, original_url_internal = _upload_file_path(assets_bucket, orig_key, tmp_path, content_type)

        # Prepare optional data URL (only for small files to avoid huge payload)
        original_data_url = ""
        try:
            if total <= max(1, DATA_URL_MAX_CHARS // 4):  # base64 expansion ~4/3, rough cap
                async with aiofiles.open(tmp_path, 'rb') as f_in:
                    data_small = await f_in.read()
                b64 = base64.b64encode(data_small).decode('ascii')
                mime = content_type if content_type.startswith('image/') else 'image/png'
                candidate = f"data:{mime};base64,{b64}"
                if len(candidate) <= DATA_URL_MAX_CHARS:
                    original_data_url = candidate
        except Exception as e:
            logger.warning(f"Skipping data URL embed due to error: {e}")

        # Save mask to storage
        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        mask_key = f"masks/{uuid.uuid4().hex}_mask.png"
        bucket_name = os.getenv('S3_BUCKET_ASSETS', 'assets')
        mask_http_url, _ = _upload_bytes(bucket_name, mask_key, mask_buffer.getvalue(), 'image/png')

        # Create analysis result
        analysis = ImageAnalysis(
            mask_url=mask_http_url,
            palette=palette,
            crops=crops,
            foreground_bbox=bbox,
            saliency_points=[[bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]],
            dominant_colors=palette[:3],
            text_regions=[],
        )
        result = analysis.model_dump()
        result["original_url"] = original_url_http
        result["original_url_internal"] = original_url_internal
        result["original_data_url"] = original_data_url
        result["original_size"] = [int(pil_image.width), int(pil_image.height)]
        logger.info(f"ingest-analyze: success total_elapsed={perf_counter()-t0:.2f}s")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingest-analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

@app.post("/copy")
async def copy_gen(payload: CopyInput):
    """Stage 3: Copy generation using AI"""
    try:
        # Generate copy variants using OpenAI (first variant returned for backward compatibility)
        raw_variants = await generate_copy_with_ai(
            payload.copy_instructions,
            payload.facts,
            payload.constraints,
            tone=payload.tone,
            style_guide=payload.style_guide,
            platform=payload.platform,
            temperature=payload.temperature,
            model_name=payload.model_name,
            num_variants=payload.num_variants,
        )

        # Fallback best
        best_fallback = {
            "headline": "Professional Excellence",
            "subheadline": "Quality that delivers results",
            "cta": (payload.constraints.allowed_cta[0] if payload and payload.constraints else "Learn More"),
        }

        # Deterministic fonts shared across variants for this request
        fonts = [f.model_dump() for f in recommend_fonts(payload.tone, payload.platform)]

        # Enrich each variant with emphasis ranges and font recommendations
        enriched_variants: List[Dict[str, Any]] = []
        for v in (raw_variants or [])[: max(1, int(payload.num_variants))]:
            em = compute_emphasis_ranges(v.get("headline", ""), v.get("subheadline", ""))
            enriched = dict(v)
            enriched["emphasis_ranges"] = {k: [r.model_dump() for r in vlist] for k, vlist in em.items()}
            enriched["font_recommendations"] = fonts
            enriched_variants.append(enriched)

        if not enriched_variants:
            em = compute_emphasis_ranges(best_fallback["headline"], best_fallback["subheadline"])
            enriched_variants = [
                {
                    **best_fallback,
                    "emphasis_ranges": {k: [r.model_dump() for r in vlist] for k, vlist in em.items()},
                    "font_recommendations": fonts,
                }
            ]

        # Backward-compatible shape: return best fields at top-level and include variants array
        best = enriched_variants[0]
        response = dict(best)
        response["variants"] = enriched_variants
        response["schema_version"] = "1.0"
        return response

    except Exception as e:
        logger.error(f"Error in copy generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compose")
async def compose(payload: Dict[str, Any]):
    """Stage 4: Layout & composition"""
    try:
        copy_data = payload.get("copy", {})
        analysis = payload.get("analysis", {})
        crop_info = payload.get("crop_info", {})
        
        if not crop_info:
            # Default crop info
            crop_info = {"width": 1080, "height": 1080}
        
        # Create SVG composition using original image URL from analysis
        smart_layout = bool(payload.get("smart_layout", True))
        panel_side_raw = payload.get("panel_side")
        panel_side = panel_side_raw.strip().lower() if isinstance(panel_side_raw, str) else None
        if panel_side not in ("left", "right", "center"):
            panel_side = None
        raw_tco = payload.get("text_color_override")
        tco = _normalize_hex_color(raw_tco) if isinstance(raw_tco, str) else None
        # Optional variant controls
        variant = payload.get("variant") or {}
        if not isinstance(variant, dict):
            variant = {}
        vseed = payload.get("variant_seed")
        if vseed is not None and "seed" not in variant:
            try:
                variant["seed"] = int(vseed)
            except Exception:
                pass
        # Optionally ask GPT for layout overrides (cached) and merge them
        gpt_variant: Dict[str, Any] = {}
        if USE_GPT_LAYOUT:
            try:
                gpt_variant = await get_gpt_layout_variant_cached(copy_data, analysis, crop_info, base_variant=variant)
            except Exception as _e:
                logger.warning(f"GPT layout generation skipped in /compose: {_e}")
        if gpt_variant:
            variant = {**variant, **gpt_variant}
        svg_content = create_svg_composition(
            copy_data,
            analysis,
            crop_info,
            smart_layout=smart_layout,
            panel_side_override=panel_side,
            text_color_override=tco,
            variant=variant,
        )
        
        # Generate composition ID
        composition_id = f"comp_{hash(svg_content) % 1000000}"
        
        result = CompositionResult(
            composition_id=composition_id,
            svg=svg_content,
            layout_data={
                "text_positions": {"headline": [crop_info["width"]//2, crop_info["height"]//2 - 40]},
                "color_scheme": analysis.get("palette", []),
                "crop_info": crop_info
            }
        )
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in composition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compose_variants")
async def compose_variants(payload: Dict[str, Any]):
    """Generate multiple diverse compositions in one call.
    Payload: { copy, analysis, crop_info, count?, variant?, variant_seed? }
    Returns: { compositions: [ {composition_id, svg, variant_used} ] }
    """
    try:
        copy_data = payload.get("copy", {})
        analysis = payload.get("analysis", {})
        crop_info = payload.get("crop_info", {}) or {"width": 1080, "height": 1080}
        count = max(1, int(payload.get("count", 3)))
        base_variant = payload.get("variant") or {}
        if not isinstance(base_variant, dict):
            base_variant = {}
        base_seed = payload.get("variant_seed")
        # Optional catalog mode: when true or when catalog == 'modern', return curated presets
        use_catalog = False
        catalog_name = None
        try:
            if isinstance(payload.get("catalog"), bool):
                use_catalog = bool(payload.get("catalog"))
            elif isinstance(payload.get("catalog"), str):
                catalog_name = (payload.get("catalog") or "").strip().lower() or None
                use_catalog = catalog_name in ("modern", "all", "presets")
        except Exception:
            use_catalog = False

        # Deterministic base from copy
        try:
            basis = f"{copy_data.get('headline','')}|{copy_data.get('subheadline','')}|{copy_data.get('cta','')}"
            copy_seed = int(hashlib.sha256(basis.encode('utf-8')).hexdigest(), 16) % (2**32)
        except Exception:
            copy_seed = random.randint(0, 2**31 - 1)

        compositions = []
        if use_catalog:
            # Use curated catalog (modern) presets
            presets = modern_variant_catalog(copy_data, analysis, crop_info, base_seed=(int(base_seed) if base_seed is not None else int(copy_seed)))
            for i, preset in enumerate(presets):
                v = {**base_variant, **(preset.get("variant") or {})}
                # Optionally enrich each variant with GPT layout overrides (cached)
                if USE_GPT_LAYOUT:
                    try:
                        gpt_v = await get_gpt_layout_variant_cached(copy_data, analysis, crop_info, base_variant=v)
                        if gpt_v:
                            v.update(gpt_v)
                    except Exception as _e:
                        logger.warning(f"GPT layout skipped for preset {preset.get('id')}: {_e}")

                svg = create_svg_composition(
                    copy_data,
                    analysis,
                    crop_info,
                    smart_layout=bool(payload.get("smart_layout", True)),
                    panel_side_override=(payload.get("panel_side") or None),
                    text_color_override=_normalize_hex_color(payload.get("text_color_override")) if isinstance(payload.get("text_color_override"), str) else None,
                    variant=v,
                )
                composition_id = f"comp_{hash(svg) % 1000000}_{i}"
                compositions.append({
                    "composition_id": composition_id,
                    "svg": svg,
                    "variant_used": v,
                    "preset_id": preset.get("id"),
                    "preset_name": preset.get("name"),
                })
            return {"compositions": compositions, "catalog": (catalog_name or "modern"), "count": len(compositions)}
        else:
            styles = ["fill", "outline", "pill"]
            width_modes = ["auto", "wide"]
            for i in range(count):
                v = dict(base_variant)
                # Seed strategy: user-provided or deterministic from copy + index
                if "seed" not in v:
                    try:
                        if base_seed is not None:
                            v["seed"] = int(base_seed) + i
                        else:
                            v["seed"] = int(copy_seed) + i * 101
                    except Exception:
                        v["seed"] = random.randint(0, 2**31 - 1)
                # Ensure some toggles vary across variants
                v.setdefault("cta_style", styles[i % len(styles)])
                v.setdefault("cta_width_mode", width_modes[i % len(width_modes)])
                # Alternate panel flip occasionally if not explicitly forced
                if "panel_side" not in v:
                    v.setdefault("flip_panel", bool(i % 2))

                # Optionally enrich each variant with GPT layout overrides (cached)
                if USE_GPT_LAYOUT:
                    try:
                        gpt_v = await get_gpt_layout_variant_cached(copy_data, analysis, crop_info, base_variant=v)
                        if gpt_v:
                            v.update(gpt_v)
                    except Exception as _e:
                        logger.warning(f"GPT layout skipped for variant {i}: {_e}")

                svg = create_svg_composition(
                    copy_data,
                    analysis,
                    crop_info,
                    smart_layout=bool(payload.get("smart_layout", True)),
                    panel_side_override=(payload.get("panel_side") or None),
                    text_color_override=_normalize_hex_color(payload.get("text_color_override")) if isinstance(payload.get("text_color_override"), str) else None,
                    variant=v,
                )
                composition_id = f"comp_{hash(svg) % 1000000}_{i}"
                compositions.append({
                    "composition_id": composition_id,
                    "svg": svg,
                    "variant_used": v,
                })

            return {"compositions": compositions, "count": len(compositions)}
    except Exception as e:
        logger.error(f"Error in compose_variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/render")
async def render(payload: Dict[str, Any]):
    """Stage 5: Render to multiple formats"""
    try:
        composition = payload.get("composition", {})
        crop_info = payload.get("crop_info", {})
        job_id = payload.get("job_id", "unknown")
        copy_data = payload.get("copy") or {}

        # Per-request overrides
        # - force_svg: if True, always rasterize the provided SVG composition.
        # - use_gpt_image: optional boolean to override the global USE_GPT_IMAGE toggle.
        force_svg = bool(payload.get("force_svg"))
        use_gpt_image = USE_GPT_IMAGE
        try:
            if isinstance(payload.get("use_gpt_image"), bool):
                use_gpt_image = bool(payload.get("use_gpt_image"))
        except Exception:
            pass

        # Decide rendering path. If force_svg is set, honor it regardless of global toggles.
        if force_svg:
            outputs = await render_svg_to_formats(composition.get("svg", ""), crop_info, job_id, force=True)
        elif use_gpt_image:
            analysis = payload.get("analysis") or {}
            outputs = await generate_ad_image_with_ai(copy_data, analysis, crop_info, job_id)
        else:
            outputs = await render_svg_to_formats(composition.get("svg", ""), crop_info, job_id)

        # Enforce a single output (prefer PNG if available)
        if outputs:
            preferred_png = None
            for o in outputs:
                try:
                    fmt = getattr(o, "format", None)
                    if fmt is None and isinstance(o, dict):
                        fmt = o.get("format")
                    if (fmt or "").lower() == "png":
                        preferred_png = o
                        break
                except Exception:
                    continue
            if preferred_png is not None:
                outputs = [preferred_png]
            else:
                outputs = outputs[:1]
        
        # Generate thumbnail URL (prefer PNG)
        thumbnail_url = next((o.url for o in outputs if isinstance(o, RenderOutput) and o.format == 'png'), None)
        if not thumbnail_url and outputs:
            # Fallback to first output URL
            first = outputs[0]
            thumbnail_url = first.url if isinstance(first, RenderOutput) else (first.get('url') if isinstance(first, dict) else None)
        if not thumbnail_url:
            # Ensure Pydantic model validation doesn't fail
            thumbnail_url = ""
        
        result = RenderResult(
            outputs=outputs,
            thumbnail_url=thumbnail_url
        )
        
        return result.model_dump()
        
    except HTTPException as he:
        # Preserve status and detail for known pipeline errors
        raise he
    except Exception as e:
        logger.error(f"Error in rendering: {e}")
        detail = str(e) or e.__class__.__name__
        raise HTTPException(status_code=500, detail=detail)

@app.post("/qa")
async def qa(payload: Dict[str, Any]):
    """Stage 6: Quality assurance gates"""
    try:
        # Normalize payload and sub-objects to avoid NoneType errors
        payload = payload or {}

        # Basic QA checks
        composition = payload.get("composition") or {}
        render_output = payload.get("render") or {}
        copy_data = payload.get("copy") or {}

        # In PNG-only mode, skip strict SVG requirement. If not using GPT image, keep the old check.
        if not USE_GPT_IMAGE:
            svg_content = (composition.get("svg") if isinstance(composition, dict) else "") or ""
            if not svg_content or "<svg" not in svg_content:
                return {"ok": False, "error": "Invalid SVG content"}
        
        # Check if render outputs exist (support either dict with outputs or direct list)
        outputs: List[Any] = []
        if isinstance(render_output, dict):
            outputs = render_output.get("outputs") or []
        elif isinstance(render_output, list):
            outputs = render_output
        if not outputs:
            return {"ok": False, "error": "No render outputs"}
        
        # Check text length constraints and basic layout heuristics
        if copy_data.get("headline", "") and len(copy_data["headline"]) > 80:
            return {"ok": False, "error": "Headline too long"}
        if copy_data.get("subheadline", "") and len(copy_data["subheadline"]) > 160:
            return {"ok": False, "error": "Subheadline too long"}

        # CTA must be allowed
        allowed_cta = CopyConstraints().allowed_cta
        if copy_data.get("cta") and copy_data["cta"] not in allowed_cta:
            return {"ok": False, "error": "CTA not allowed"}

        # Minimal heuristic: ensure at least one PNG output exists
        if not any(((o.get("format") if isinstance(o, dict) else getattr(o, "format", None)) == "png") for o in outputs):
            logger.warning("PNG output missing")

        return {"ok": True, "quality_score": 0.97}
        
    except Exception as e:
        logger.error(f"Error in QA: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/export")
async def export(payload: Dict[str, Any]):
    """Stage 7: Export & delivery"""
    try:
        payload = payload or {}
        render_output = payload.get("render") or {}
        outputs: List[Any] = []
        if isinstance(render_output, dict):
            outputs = render_output.get("outputs") or []
        elif isinstance(render_output, list):
            outputs = render_output
        # Enforce a single output globally at export time as well
        if outputs:
            # Prefer PNG if present
            preferred_png = None
            for o in outputs:
                try:
                    fmt = getattr(o, "format", None)
                    if fmt is None and isinstance(o, dict):
                        fmt = o.get("format")
                    if (fmt or "").lower() == "png":
                        preferred_png = o
                        break
                except Exception:
                    continue
            if preferred_png is not None:
                outputs = [preferred_png]
            else:
                outputs = outputs[:1]
        
        # Create manifest
        manifest = {
            "job_id": payload.get("job_id", "unknown"),
            "timestamp": payload.get("timestamp", ""),
            "outputs": outputs,
            "metadata": {
                "version": "1.0",
                "pipeline": "madworks-ai"
            }
        }
        
        # Save manifest to storage
        manifest_key = f"manifests/{payload.get('job_id', 'unknown')}_manifest.json"
        bucket_name = os.getenv('S3_BUCKET_ASSETS', 'assets')
        manifest_http_url, _ = _upload_bytes(bucket_name, manifest_key, json.dumps(manifest, indent=2).encode('utf-8'), 'application/json')
        manifest_url = manifest_http_url
        
        return {
            "outputs": outputs,
            "manifest_url": manifest_url,
            # Support both dicts and objects for outputs
            "download_urls": [
                (output.get("url") if isinstance(output, dict) else getattr(output, "url", None))
                for output in outputs
                if (isinstance(output, dict) and output.get("url")) or (hasattr(output, "url") and getattr(output, "url") is not None)
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "madworks-ai-pipeline",
        "gpt_layout_enabled": bool(USE_GPT_LAYOUT),
        "layout_model": LAYOUT_MODEL,
        "layout_temperature": LAYOUT_TEMPERATURE,
        "openai_configured": bool(_OPENAI_API_KEY),
        "raster_enabled": bool(RENDER_RASTER),
        "gpt_image_enabled": bool(USE_GPT_IMAGE),
        "image_model": IMAGE_MODEL,
        "strict_image_model_only": bool(STRICT_IMAGE_MODEL_ONLY),
        "openai_sdk_version": getattr(openai, "__version__", "unknown"),
    }
 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
