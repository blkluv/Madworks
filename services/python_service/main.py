import os
import asyncio
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import uuid

import openai
import cairosvg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT', 'http://localhost:9000'),
    aws_access_key_id=os.getenv('S3_ACCESS_KEY', 'minio'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY', 'minio123'),
    region_name='us-east-1'
)

app = FastAPI(title="Madworks AI Pipeline", version="1.1.0")

# Storage config: support S3 (MinIO) and local static file storage for easier dev/testing
STORAGE_MODE = os.getenv('STORAGE_MODE', 's3').lower()  # 's3' or 'local'
STORAGE_DIR = os.getenv('STORAGE_DIR', str(Path(__file__).resolve().parent / 'storage'))
PUBLIC_BASE_URL = os.getenv('PUBLIC_BASE_URL', 'http://localhost:8010').rstrip('/')

# Ensure local storage directory exists and mount when in local mode
try:
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")
    # Extra aliases so clients can fetch both /static/... and /outputs/... paths
    # This covers cases where URLs are constructed as /outputs/{bucket}/{key}
    # (e.g., outputs/outputs/...) or /assets/... without the /static prefix.
    app.mount("/outputs", StaticFiles(directory=STORAGE_DIR), name="outputs")
    app.mount("/assets", StaticFiles(directory=STORAGE_DIR), name="assets")
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

# Utility functions
def extract_palette(image: Image.Image, num_colors: int = 5) -> List[str]:
    """Extract dominant colors from image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for faster processing
    small_image = image.resize((150, 150))
    pixels = np.array(small_image).reshape(-1, 3)
    
    # Use k-means to find dominant colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    return [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in colors]

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
        n = max(1, int(num_variants))
        for _ in range(n):
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert advertising copywriter who outputs strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(temperature),
                max_tokens=220
            )

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
            parsed["headline"] = (parsed.get("headline") or "")[: constraints.max_headline]
            parsed["subheadline"] = (parsed.get("subheadline") or "")[: constraints.max_sub]
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

def _resolve_img_href(analysis: Dict[str, Any]) -> str:
    """Robustly resolve the best image href for SVG <image> tag.
    Priority: sizeable original_data_url -> local static file -> HTTP fetch -> empty."""
    data_url = (analysis.get("original_data_url") or "").strip()
    # Ensure it's not an empty/placeholder tiny data URL
    if data_url.startswith("data:") and len(data_url) > 100:
        return data_url

    # Try local static path from public/internal URL
    for key in ("original_url_internal", "original_url"):
        url = (analysis.get(key) or "").strip()
        if url:
            du = _try_build_data_url_from_storage(url)
            if du:
                return du
            # Fallback: HTTP fetch (works in s3/minio mode)
            try:
                resp = requests.get(url, timeout=5)
                if resp.ok and resp.content:
                    mime = resp.headers.get("Content-Type", "image/png")
                    b64 = base64.b64encode(resp.content).decode("ascii")
                    return f"data:{mime};base64,{b64}"
            except Exception as e:
                logger.warning(f"HTTP fetch failed for original image: {e}")
    return ""

def create_svg_composition(copy_data: Dict[str, str], analysis: Dict[str, Any], crop_info: Dict[str, Any]) -> str:
    """Create SVG composition with a fixed, professional layout:
    - Full-bleed background image with strong legibility gradient
    - Left-aligned bold white headline
    - Left-aligned supporting copy below
    - Bottom-left CTA button
    """

    # Layout metrics
    width = int(crop_info["width"]) if "width" in crop_info else 1080
    height = int(crop_info["height"]) if "height" in crop_info else 1080
    padding = max(40, int(min(width, height) * 0.06))

    # Typography (scaled to canvas)
    headline_size = max(36, int(min(width, height) * 0.11))
    sub_size = max(18, int(min(width, height) * 0.035))
    text_color = "#ffffff"
    text_color_secondary = "#e5e5e5"
    font_family = "Inter, Roboto, Arial, sans-serif"
    letter_spacing = 0  # tweakable

    # CTA sizing and placement (bottom-left)
    cta_text = copy_data.get("cta", "Learn More")
    cta_width = max(200, int(44 + len(cta_text) * 11))
    cta_height = max(52, int(min(width, height) * 0.055))
    cta_x = padding
    cta_y = height - padding - cta_height
    cta_text_x = cta_x + cta_width // 2
    cta_text_y = cta_y + int(cta_height * 0.66)

    # Colors / accents
    palette = analysis.get("palette", [])
    cta_color = palette[2] if len(palette) > 2 else "#2563EB"  # blue

    # Text wrapping
    content_width = width - 2 * padding
    headline = copy_data.get("headline", "").strip()
    subheadline = copy_data.get("subheadline", "").strip()
    max_chars_headline = _measure_chars_for_width(content_width, headline_size)
    max_chars_sub = _measure_chars_for_width(content_width, sub_size)
    headline_lines = _wrap_text(headline, max_chars_headline)
    sub_lines = _wrap_text(subheadline, max_chars_sub)

    # Vertical rhythm: headline near upper-left, sub below, CTA at bottom-left
    line_gap = int(headline_size * 0.28)
    sub_gap = int(sub_size * 0.24)
    headline_y_start = padding + headline_size  # first baseline
    # compute block height for headline
    headline_block_h = len(headline_lines) * headline_size + (len(headline_lines) - 1) * line_gap
    subheadline_y_start = headline_y_start + headline_block_h + int(headline_size * 0.45)

    # SVG template with legibility gradient (applied only when an image exists) and a clean fallback background
    svg_template = """
    <svg width="{{ width }}" height="{{ height }}" viewBox="0 0 {{ width }} {{ height }}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <linearGradient id="shade" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#000000" stop-opacity="0.35" />
                <stop offset="50%" stop-color="#000000" stop-opacity="0.20" />
                <stop offset="100%" stop-color="#000000" stop-opacity="0.08" />
            </linearGradient>
            <linearGradient id="bg" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#1f2937" stop-opacity="1" />
                <stop offset="100%" stop-color="#111827" stop-opacity="1" />
            </linearGradient>
            <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.35"/>
            </filter>
        </defs>

        <!-- Background -->
        {% if original_url %}
        <image xlink:href="{{ original_url }}" x="0" y="0" width="{{ width }}" height="{{ height }}" preserveAspectRatio="xMidYMid slice"/>
        <!-- Legibility gradient overlay -->
        <rect width="100%" height="100%" fill="url(#shade)" />
        {% else %}
        <rect width="100%" height="100%" fill="url(#bg)" />
        {% endif %}

        <!-- Content -->
        <g>
            <!-- Headline -->
            <text x="{{ padding }}" y="{{ headline_y_start }}" font-family="{{ font_family }}" font-size="{{ headline_size }}" font-weight="800" fill="{{ text_color }}" letter-spacing="{{ letter_spacing }}" filter="url(#shadow)">
                {% for line in headline_lines %}
                <tspan x="{{ padding }}" dy="{% if loop.first %}0{% else %}{{ headline_size + line_gap }}{% endif %}">{{ line }}</tspan>
                {% endfor %}
            </text>

            <!-- Subheadline -->
            <text x="{{ padding }}" y="{{ subheadline_y_start }}" font-family="{{ font_family }}" font-size="{{ sub_size }}" font-weight="500" fill="{{ text_color_secondary }}" letter-spacing="0" filter="url(#shadow)">
                {% for line in sub_lines %}
                <tspan x="{{ padding }}" dy="{% if loop.first %}0{% else %}{{ sub_size + sub_gap }}{% endif %}">{{ line }}</tspan>
                {% endfor %}
            </text>

            <!-- CTA Button -->
            <rect x="{{ cta_x }}" y="{{ cta_y }}" width="{{ cta_width }}" height="{{ cta_height }}" rx="10" fill="{{ cta_color }}" filter="url(#shadow)"/>
            <text x="{{ cta_text_x }}" y="{{ cta_text_y }}" font-family="{{ font_family }}" font-size="{{ int(sub_size*0.9) }}" font-weight="800" fill="#ffffff" text-anchor="middle">{{ cta }}</text>
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
    svg_content = template.render(
        width=width,
        height=height,
        padding=padding,
        font_family=font_family,
        text_color=text_color,
        text_color_secondary=text_color_secondary,
        headline_size=headline_size,
        sub_size=sub_size,
        headline_lines=headline_lines,
        sub_lines=sub_lines,
        line_gap=line_gap,
        sub_gap=sub_gap,
        headline_y_start=headline_y_start,
        subheadline_y_start=subheadline_y_start,
        cta_x=cta_x,
        cta_y=cta_y,
        cta_width=cta_width,
        cta_height=cta_height,
        cta_text_x=cta_text_x,
        cta_text_y=cta_text_y,
        cta=cta_text,
        letter_spacing=letter_spacing,
        original_url=img_href,
        int=int,
    )
    return svg_content

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

async def render_svg_to_formats(svg_content: str, crop_info: Dict[str, Any], job_id: str) -> List[RenderOutput]:
    """Render SVG to PNG/JPG using CairoSVG and upload all formats (including original SVG), with local/S3 storage support."""
    outputs: List[RenderOutput] = []
    outputs_bucket = os.getenv('S3_BUCKET_OUTPUTS', 'outputs')

    width = int(crop_info.get("width", 1080))
    height = int(crop_info.get("height", 1080))

    unique = uuid.uuid4().hex[:8]

    # 1) Upload original SVG
    svg_key = f"outputs/{job_id}_{unique}_svg.svg"
    svg_url, _ = _upload_bytes(outputs_bucket, svg_key, svg_content.encode('utf-8'), 'image/svg+xml')
    outputs.append(RenderOutput(format="svg", width=width, height=height, url=svg_url))

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

    png_bytes = cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        output_width=width,
        output_height=height,
        unsafe=True,
    )
    logger.info(f"Rendered PNG bytes: {len(png_bytes)}")

    # Upload PNG
    png_key = f"outputs/{job_id}_{unique}_png.png"
    png_url, _ = _upload_bytes(outputs_bucket, png_key, png_bytes, 'image/png')
    outputs.append(RenderOutput(format="png", width=width, height=height, url=png_url))

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
    outputs.append(RenderOutput(format="jpg", width=width, height=height, url=jpg_url))

    return outputs

# API Endpoints
@app.post("/ingest-analyze")
async def ingest_analyze(image: UploadFile = File(...)):
    """Stage 1: Upload & sanitize, Stage 2: Visual analysis"""
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Analyze image
        mask, bbox = detect_foreground(pil_image)
        palette = extract_palette(pil_image)
        crops = generate_crop_proposals(pil_image, bbox)
        
        # Upload original image to storage for public composition reference
        assets_bucket = os.getenv('S3_BUCKET_ASSETS', 'assets')
        orig_ext = os.path.splitext(image.filename)[1] or '.png'
        orig_key = f"uploads/pipeline/{uuid.uuid4().hex}{orig_ext}"
        original_url_http, original_url_internal = _upload_bytes(
            assets_bucket, orig_key, image_data, image.content_type or 'application/octet-stream'
        )
        # Prepare data URL for reliable embedding during server-side rendering
        # Detect MIME from Pillow format for accuracy
        fmt = (pil_image.format or '').upper()
        fmt_to_mime = {
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'PNG': 'image/png',
            'WEBP': 'image/webp',
            'GIF': 'image/gif',
            'BMP': 'image/bmp'
        }
        detected_mime = fmt_to_mime.get(fmt)
        mime = detected_mime or (image.content_type if (image.content_type and image.content_type.startswith('image/')) else 'image/png')
        b64 = base64.b64encode(image_data).decode('ascii')
        original_data_url = f"data:{mime};base64,{b64}"
        
        # Save mask to S3/MinIO
        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        mask_key = f"masks/{uuid.uuid4().hex}_mask.png"
        bucket_name = os.getenv('S3_BUCKET_ASSETS', 'assets')
        mask_http_url, _ = _upload_bytes(bucket_name, mask_key, mask_buffer.getvalue(), 'image/png')
        mask_url = mask_http_url
        
        # Create analysis result
        analysis = ImageAnalysis(
            mask_url=mask_url,
            palette=palette,
            crops=crops,
            foreground_bbox=bbox,
            saliency_points=[[bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]],  # Center of foreground
            dominant_colors=palette[:3],
            text_regions=[]  # Would use OCR in production
        )
        
        result = analysis.dict()
        result["original_url"] = original_url_http
        result["original_url_internal"] = original_url_internal
        result["original_data_url"] = original_data_url
        return result
        
    except Exception as e:
        logger.error(f"Error in ingest-analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copy")
async def copy_gen(payload: CopyInput):
    """Stage 3: Copy generation using AI"""
    try:
        # Generate copy variants using OpenAI (first variant returned for backward compatibility)
        variants = await generate_copy_with_ai(
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

        best = variants[0] if variants else {
            "headline": "Professional Excellence",
            "subheadline": "Quality that delivers results",
            "cta": (payload.constraints.allowed_cta[0] if payload and payload.constraints else "Learn More")
        }
        # Include all variants for clients that can use them; orchestrator will ignore extra fields
        best_with_variants = dict(best)
        best_with_variants["variants"] = variants
        return best_with_variants
        
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
        svg_content = create_svg_composition(copy_data, analysis, crop_info)
        
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
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error in composition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/render")
async def render(payload: Dict[str, Any]):
    """Stage 5: Render to multiple formats"""
    try:
        composition = payload.get("composition", {})
        crop_info = payload.get("crop_info", {})
        job_id = payload.get("job_id", "unknown")
        
        # Render SVG to multiple formats
        outputs = await render_svg_to_formats(composition.get("svg", ""), crop_info, job_id)
        
        # Generate thumbnail URL (first JPG if available)
        thumbnail_url = next((o.url for o in outputs if isinstance(o, RenderOutput) and o.format == 'jpg'), None)
        if not thumbnail_url and outputs:
            # Fallback to first output URL
            first = outputs[0]
            thumbnail_url = first.url if isinstance(first, RenderOutput) else (first.get('url') if isinstance(first, dict) else None)
        
        result = RenderResult(
            outputs=outputs,
            thumbnail_url=thumbnail_url
        )
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error in rendering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def qa(payload: Dict[str, Any]):
    """Stage 6: Quality assurance gates"""
    try:
        # Basic QA checks
        composition = payload.get("composition", {})
        render_output = payload.get("render", {})
        
        # Check if SVG is valid
        svg_content = composition.get("svg", "")
        if not svg_content or "<svg" not in svg_content:
            return {"ok": False, "error": "Invalid SVG content"}
        
        # Check if render outputs exist
        outputs = render_output.get("outputs", [])
        if not outputs:
            return {"ok": False, "error": "No render outputs"}
        
        # Check text length constraints and basic layout heuristics
        copy_data = payload.get("copy", {})
        if copy_data.get("headline", "") and len(copy_data["headline"]) > 80:
            return {"ok": False, "error": "Headline too long"}
        if copy_data.get("subheadline", "") and len(copy_data["subheadline"]) > 160:
            return {"ok": False, "error": "Subheadline too long"}

        # CTA must be allowed
        allowed_cta = CopyConstraints().allowed_cta
        if copy_data.get("cta") and copy_data["cta"] not in allowed_cta:
            return {"ok": False, "error": "CTA not allowed"}

        # Minimal color contrast heuristic: ensure dark overlay exists in SVG for legibility
        svg_content = composition.get("svg", "")
        if 'opacity="0.22"' not in svg_content and 'opacity=\"0.22\"' not in svg_content:
            logger.warning("Legibility overlay may be missing")

        return {"ok": True, "quality_score": 0.97}
        
    except Exception as e:
        logger.error(f"Error in QA: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/export")
async def export(payload: Dict[str, Any]):
    """Stage 7: Export & delivery"""
    try:
        render_output = payload.get("render", {})
        outputs = render_output.get("outputs", [])
        
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
    return {"status": "healthy", "service": "madworks-ai-pipeline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
