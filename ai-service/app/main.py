"""FastAPI entrypoint. POST /analyze -> full pipeline."""
from __future__ import annotations

import io
import json
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

# Register HEIC/HEIF (iPhone photos) so Pillow can open them.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pillow-heif optional; JPEG/PNG still work without it
    pass

from .detect import make_detector, table_pockets
from .geometry import analyze_shots
from .models import AnalyzeResult
from .overlay import draw_overlay, encode_png
from .table import OUT_H, OUT_W, detect_table

app = FastAPI(title="Pool Vision AI - AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = make_detector()


def decode_image(raw: bytes) -> np.ndarray | None:
    """Decode upload to a BGR ndarray.

    Tries OpenCV first; on failure falls back to Pillow, which (with
    pillow-heif) handles HEIC/HEIF iPhone photos and other formats OpenCV
    cannot. EXIF orientation is applied so sideways phone photos sit upright.
    """
    if not raw:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


def _parse_corners(corners: Optional[str]) -> Optional[List[List[float]]]:
    """Parse the optional `corners` form field: JSON [[x,y]*4] normalized.

    Corners are TL, TR, BR, BL with x,y in [0,1] (values outside allowed for
    off-frame/occluded corners). Returns None when absent; raises 400 on a
    malformed value so the client gets a clear error rather than silent
    auto-detect fallback.
    """
    if not corners:
        return None
    try:
        pts = json.loads(corners)
        if (
            isinstance(pts, list)
            and len(pts) == 4
            and all(len(p) == 2 for p in pts)
        ):
            return [[float(p[0]), float(p[1])] for p in pts]
    except (ValueError, TypeError):
        pass
    raise HTTPException(
        status_code=400, detail="corners must be JSON [[x,y],...] of 4 points"
    )


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(
    image: UploadFile = File(...),
    corners: Optional[str] = Form(None),
    target_ball: Optional[int] = Form(None),
):
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    img = decode_image(raw)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="could not decode image (unsupported format?)",
        )

    warped, table_detected = detect_table(img, _parse_corners(corners))
    balls = detector.detect(warped)
    pockets = table_pockets()
    cue = next((b for b in balls if b.is_cue), None)
    best, candidates = analyze_shots(cue, balls, pockets, target_ball)
    overlay_png = draw_overlay(warped, balls, pockets, best)

    return AnalyzeResult(
        table_detected=table_detected,
        width=OUT_W,
        height=OUT_H,
        cue_ball=cue,
        balls=balls,
        pockets=pockets,
        shot=best,
        candidates=candidates,
        overlay_png=overlay_png,
        warped_png=encode_png(warped),
    )
