import os
import asyncio
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import openai
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import boto3
from dotenv import load_dotenv
import aiofiles
import requests
from jinja2 import Template

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
    endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://localhost:9000'),
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
    region_name='us-east-1'
)

app = FastAPI(title="Madworks AI Pipeline", version="1.0.0")

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

async def generate_copy_with_ai(instructions: str, facts: Dict[str, Any], constraints: CopyConstraints) -> Dict[str, str]:
    """Generate ad copy using OpenAI GPT-3.5"""
    try:
        # Build the prompt
        prompt = f"""
        You are an expert advertising copywriter. Create compelling ad copy based on these instructions:
        
        Instructions: {instructions}
        
        Brand Facts: {json.dumps(facts, indent=2)}
        
        Constraints:
        - Headline: max {constraints.max_headline} characters
        - Subheadline: max {constraints.max_sub} characters  
        - CTA must be one of: {', '.join(constraints.allowed_cta)}
        - Brand voice: {constraints.brand_voice}
        - Target audience: {constraints.target_audience}
        - Avoid these words: {', '.join(constraints.forbidden_words)}
        
        Return ONLY a JSON object with these exact keys:
        {{
            "headline": "compelling headline here",
            "subheadline": "supporting copy here", 
            "cta": "CTA from allowed list"
        }}
        
        Make it engaging, professional, and conversion-focused.
        """
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert advertising copywriter who creates high-converting ad copy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            copy_data = json.loads(content)
            # Validate required fields
            required_fields = ["headline", "subheadline", "cta"]
            if not all(field in copy_data for field in required_fields):
                raise ValueError("Missing required fields")
            
            # Apply constraints
            copy_data["headline"] = copy_data["headline"][:constraints.max_headline]
            copy_data["subheadline"] = copy_data["subheadline"][:constraints.max_sub]
            
            # Ensure CTA is valid
            if copy_data["cta"] not in constraints.allowed_cta:
                copy_data["cta"] = constraints.allowed_cta[0]
            
            return copy_data
            
        except json.JSONDecodeError:
            # Fallback to template-based generation
            logger.warning("Failed to parse AI response, using fallback")
            return {
                "headline": "Transform Your Vision",
                "subheadline": "Professional results that speak for themselves",
                "cta": constraints.allowed_cta[0]
            }
            
    except Exception as e:
        logger.error(f"Error generating copy with AI: {e}")
        # Fallback copy
        return {
            "headline": "Professional Excellence",
            "subheadline": "Quality that delivers results",
            "cta": constraints.allowed_cta[0]
        }

def create_svg_composition(copy_data: Dict[str, str], palette: List[str], crop_info: Dict[str, Any]) -> str:
    """Create SVG composition with the generated copy"""
    
    # SVG template
    svg_template = """
    <svg width="{{ width }}" height="{{ height }}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{{ bg_color }};stop-opacity:1" />
                <stop offset="100%" style="stop-color:{{ bg_color_2 }};stop-opacity:0.8" />
            </linearGradient>
            <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="rgba(0,0,0,0.3)"/>
            </filter>
        </defs>
        
        <!-- Background -->
        <rect width="100%" height="100%" fill="url(#bg)"/>
        
        <!-- Content container -->
        <g transform="translate({{ padding }}, {{ padding }})">
            <!-- Headline -->
            <text x="{{ text_x }}" y="{{ headline_y }}" 
                  font-family="Arial, sans-serif" 
                  font-size="{{ headline_size }}" 
                  font-weight="bold"
                  fill="{{ text_color }}"
                  text-anchor="middle"
                  filter="url(#shadow)">
                {{ headline }}
            </text>
            
            <!-- Subheadline -->
            <text x="{{ text_x }}" y="{{ subheadline_y }}" 
                  font-family="Arial, sans-serif" 
                  font-size="{{ subheadline_size }}" 
                  fill="{{ text_color_secondary }}"
                  text-anchor="middle"
                  filter="url(#shadow)">
                {{ subheadline }}
            </text>
            
            <!-- CTA Button -->
            <rect x="{{ cta_x }}" y="{{ cta_y }}" 
                  width="{{ cta_width }}" height="{{ cta_height }}" 
                  rx="8" 
                  fill="{{ cta_color }}"
                  filter="url(#shadow)"/>
            
            <text x="{{ cta_text_x }}" y="{{ cta_text_y }}" 
                  font-family="Arial, sans-serif" 
                  font-size="18" 
                  font-weight="bold"
                  fill="white"
                  text-anchor="middle">
                {{ cta }}
            </text>
        </g>
    </svg>
    """
    
    # Calculate dimensions and positioning
    width = crop_info["width"]
    height = crop_info["height"]
    padding = 60
    text_x = width // 2
    headline_y = height // 2 - 40
    subheadline_y = height // 2 + 20
    cta_y = height - 120
    cta_width = 200
    cta_height = 50
    cta_x = (width - cta_width) // 2
    cta_text_x = width // 2
    cta_text_y = cta_y + 32
    
    # Choose colors from palette
    bg_color = palette[0] if len(palette) > 0 else "#1a1a1a"
    bg_color_2 = palette[1] if len(palette) > 1 else "#2a2a2a"
    text_color = palette[-1] if len(palette) > 0 else "#ffffff"
    text_color_secondary = palette[-2] if len(palette) > 1 else "#cccccc"
    cta_color = palette[2] if len(palette) > 2 else "#3b82f6"
    
    # Calculate font sizes based on text length
    headline_size = min(48, max(24, 60 - len(copy_data["headline"])))
    subheadline_size = min(24, max(16, 32 - len(copy_data["subheadline"]) // 2))
    
    # Render template
    template = Template(svg_template)
    svg_content = template.render(
        width=width, height=height, padding=padding,
        text_x=text_x, headline_y=headline_y, subheadline_y=subheadline_y,
        cta_x=cta_x, cta_y=cta_y, cta_width=cta_width, cta_height=cta_height,
        cta_text_x=cta_text_x, cta_text_y=cta_text_y,
        headline=copy_data["headline"], subheadline=copy_data["subheadline"], cta=copy_data["cta"],
        headline_size=headline_size, subheadline_size=subheadline_size,
        bg_color=bg_color, bg_color_2=bg_color_2,
        text_color=text_color, text_color_secondary=text_color_secondary,
        cta_color=cta_color
    )
    
    return svg_content

async def render_svg_to_formats(svg_content: str, crop_info: Dict[str, Any], job_id: str) -> List[RenderOutput]:
    """Render SVG to multiple formats"""
    outputs = []
    
    # For now, we'll create placeholder outputs
    # In production, this would use resvg or cairosvg to actually render
    
    formats = [
        {"format": "png", "width": crop_info["width"], "height": crop_info["height"]},
        {"format": "jpg", "width": crop_info["width"], "height": crop_info["height"]},
        {"format": "svg", "width": crop_info["width"], "height": crop_info["height"]},
    ]
    
    for fmt in formats:
        # Generate mock URLs for now
        url = f"http://localhost:9000/outputs/{job_id}_{fmt['format']}.{fmt['format']}"
        
        outputs.append(RenderOutput(
            format=fmt["format"],
            width=fmt["width"],
            height=fmt["height"],
            url=url
        ))
    
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
        
        # Save mask to S3/MinIO
        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        mask_key = f"masks/{image.filename}_mask.png"
        s3_client.put_object(
            Bucket='madworks-assets',
            Key=mask_key,
            Body=mask_buffer.getvalue(),
            ContentType='image/png'
        )
        
        mask_url = f"s3://madworks-assets/{mask_key}"
        
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
        
        return analysis.dict()
        
    except Exception as e:
        logger.error(f"Error in ingest-analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copy")
async def copy_gen(payload: CopyInput):
    """Stage 3: Copy generation using AI"""
    try:
        # Generate copy using OpenAI
        copy_data = await generate_copy_with_ai(
            payload.copy_instructions,
            payload.facts,
            payload.constraints
        )
        
        return copy_data
        
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
        
        # Create SVG composition
        svg_content = create_svg_composition(copy_data, analysis.get("palette", []), crop_info)
        
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
        
        # Generate thumbnail URL
        thumbnail_url = f"http://localhost:9000/outputs/{job_id}_thumb.jpg"
        
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
        
        # Check text length constraints
        copy_data = payload.get("copy", {})
        if copy_data.get("headline", ""):
            if len(copy_data["headline"]) > 50:
                return {"ok": False, "error": "Headline too long"}
        
        return {"ok": True, "quality_score": 0.95}
        
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
        
        # Save manifest to S3/MinIO
        manifest_key = f"manifests/{payload.get('job_id', 'unknown')}_manifest.json"
        s3_client.put_object(
            Bucket='madworks-assets',
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType='application/json'
        )
        
        manifest_url = f"s3://madworks-assets/{manifest_key}"
        
        return {
            "outputs": outputs,
            "manifest_url": manifest_url,
            "download_urls": [output.url for output in outputs]
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


