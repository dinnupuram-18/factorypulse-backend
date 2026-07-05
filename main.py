"""
BottleGuard AI — Inference API

Serves a trained YOLOv8 model (best.pt) to detect bottle defects.

MOCK MODE: If no trained weights are found at MODEL_PATH, this service automatically
falls back to returning realistic-looking mock detections instead of crashing. This lets
you build and test the whole frontend tonight, before training finishes tomorrow.
Once best.pt exists and MODEL_PATH points to it, real inference kicks in automatically —
no code changes needed, no frontend changes needed.
"""

import io
import os
import random
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
CLASS_NAMES = ["broken_large", "broken_small", "contamination"]

app = FastAPI(title="BottleGuard AI Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin before real prod use
    allow_methods=["*"],
    allow_headers=["*"],
)


class Defect(BaseModel):
    cls: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] in original image pixel coordinates


class InspectResponse(BaseModel):
    defects: List[Defect]
    inspectionTimeMs: float
    imageWidth: int
    imageHeight: int
    mode: str  # "trained" or "mock" — frontend can show this honestly if you want


# --- Model loading, with graceful mock fallback ---
_model = None
_mode = "mock"

def _load_model():
    global _model, _mode
    path = Path(MODEL_PATH)
    if not path.exists():
        print(f"[BottleGuard] No weights found at '{MODEL_PATH}'. Running in MOCK mode.")
        _mode = "mock"
        return
    try:
        from ultralytics import YOLO
        _model = YOLO(str(path))
        _mode = "trained"
        print(f"[BottleGuard] Loaded trained weights from '{MODEL_PATH}'. Running in TRAINED mode.")
    except Exception as e:
        print(f"[BottleGuard] Failed to load weights ({e}). Falling back to MOCK mode.")
        _mode = "mock"


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "mode": _mode}


def _mock_predict(img_w: int, img_h: int) -> List[Defect]:
    """
    Deterministic-ish mock defect generator, roughly centered with plausible box sizes,
    so the frontend overlay looks realistic during development.
    """
    # ~30% chance of a "clean" bottle with no defects, like a real inspection would see
    if random.random() < 0.3:
        return []

    n_defects = random.choice([1, 1, 1, 2])
    defects = []
    for _ in range(n_defects):
        cls = random.choice(CLASS_NAMES)
        box_w = random.uniform(0.08, 0.22) * img_w
        box_h = random.uniform(0.08, 0.22) * img_h
        x1 = random.uniform(0.1, 0.7) * img_w
        y1 = random.uniform(0.1, 0.7) * img_h
        defects.append(
            Defect(
                cls=cls,
                confidence=round(random.uniform(0.68, 0.97), 3),
                box=[round(x1, 1), round(y1, 1), round(x1 + box_w, 1), round(y1 + box_h, 1)],
            )
        )
    return defects


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    img_w, img_h = img.size
    start = time.perf_counter()

    if _mode == "trained" and _model is not None:
        results = _model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        defects = []
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            defects.append(Defect(cls=cls_name, confidence=round(conf, 3), box=[x1, y1, x2, y2]))
    else:
        defects = _mock_predict(img_w, img_h)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return InspectResponse(
        defects=defects,
        inspectionTimeMs=round(elapsed_ms, 1),
        imageWidth=img_w,
        imageHeight=img_h,
        mode=_mode,
    )
