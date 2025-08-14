from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

app = FastAPI(title="Madworks Pipeline Stub")


class CopyConstraints(BaseModel):
    max_headline: int = 38
    max_sub: int = 80
    allowed_cta: List[str] = ["Shop Now", "Learn More", "Try it now"]
    forbidden_words: List[str] = ["guarantee", "#1", "best ever"]


class CopyInput(BaseModel):
    copy: Dict[str, Any] = {}
    facts: Dict[str, Any] = {}
    brand_kit_id: Optional[str] = None
    constraints: CopyConstraints = Field(default_factory=CopyConstraints)


@app.post("/ingest-analyze")
def ingest_analyze(payload: Dict[str, Any]):
    return {
        "mask_url": "s3://assets/mask.png",
        "palette": ["#111111", "#222222", "#eeeeee"],
        "crops": [{"ratio": "1:1"}, {"ratio": "4:5"}, {"ratio": "9:16"}],
    }


@app.post("/copy")
def copy_gen(payload: CopyInput):
    provided = payload.copy or {}
    headline = provided.get("headline") or "Sample Headline"
    sub = provided.get("sub") or "Short supporting copy"
    cta = provided.get("cta") or "Learn More"

    # guardrails
    for bad in payload.constraints.forbidden_words:
        headline = headline.replace(bad, "")
        sub = sub.replace(bad, "")

    if cta not in payload.constraints.allowed_cta:
        cta = payload.constraints.allowed_cta[0]

    headline = headline[: payload.constraints.max_headline]
    sub = sub[: payload.constraints.max_sub]

    return {"headline": headline, "sub": sub, "cta": cta}


@app.post("/compose")
def compose(payload: Dict[str, Any]):
    return {"composition_id": "cmp_123", "svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>"}


@app.post("/render")
def render(payload: Dict[str, Any]):
    outputs = [
        {"format": "png", "w": 1080, "h": 1350, "url": "http://localhost:9000/outputs/portrait.png"},
        {"format": "png", "w": 1080, "h": 1080, "url": "http://localhost:9000/outputs/square.png"},
        {"format": "jpg", "w": 1920, "h": 1080, "url": "http://localhost:9000/outputs/landscape.jpg"},
        {"format": "pdf", "w": 2550, "h": 3300, "url": "http://localhost:9000/outputs/print.pdf"},
        {"format": "svg", "w": 1080, "h": 1080, "url": "http://localhost:9000/outputs/square.svg"},
    ]
    return {"outputs": outputs, "thumbnail_url": "http://localhost:9000/outputs/thumb.jpg"}


@app.post("/qa")
def qa(_: Dict[str, Any]):
    return {"ok": True}


@app.post("/export")
def export(payload: Dict[str, Any]):
    manifest_url = "http://localhost:9000/outputs/manifest.json"
    outputs = payload.get("render", {}).get("outputs", []) or [
        {"format": "png", "w": 1080, "h": 1080, "url": "http://localhost:9000/outputs/square.png"}
    ]
    return {"outputs": outputs, "manifest_url": manifest_url}


