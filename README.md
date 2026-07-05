# FactoryPulse — Inference Backend

FastAPI service that serves a YOLOv8 object-detection model trained to spot manufacturing
defects on glass bottles. Built as part of a solo hackathon slice of a broader
"Factory Intelligence OS" vision — this repo covers the vision-QC piece specifically.

## What it does

- Trained on the MVTec AD "bottle" category — 3 defect classes: `broken_large`,
  `broken_small`, `contamination`
- One endpoint, `POST /inspect`: send an image, get back defect boxes, class, and confidence
- Automatically falls back to mock detections if no trained weights (`best.pt`) are found,
  so the rest of the app can be built/demoed before training finishes
- CPU inference — no GPU needed to serve requests

## Real trained metrics (validation set)

| Metric | Value |
|---|---|
| Overall precision | 0.705 |
| Overall recall | 0.333 |
| Overall mAP50 | 0.515 |
| Overall mAP50-95 | 0.347 |

Per-class performance varies — `broken_large` and `broken_small` detect well; `contamination`
is weaker, likely due to a very small validation sample for that class. This is an honest
limitation of the small training set (~292 images total), not a pipeline flaw.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Place a trained `best.pt` in this folder to run real inference. Without it, the service
runs in mock mode automatically. Check `/health` to see which mode is active.

## Endpoints

- `GET /health` — service status and current mode (`trained` or `mock`)
- `POST /inspect` — accepts an image file, returns:
  ```json
  {
    "defects": [{ "cls": "broken_large", "confidence": 0.87, "box": [x1, y1, x2, y2] }],
    "inspectionTimeMs": 181,
    "imageWidth": 900,
    "imageHeight": 900,
    "mode": "trained"
  }
  ```

## Scope

Single object category (bottle), single camera view. Multi-object/multi-camera factory
coverage, digital twin, predictive maintenance, and OEE analytics are future scope — not
built here. See the frontend repo for a simulated preview of where this is headed.

## Stack

FastAPI · Ultralytics YOLOv8 · Pydantic · deployed on Render/Railway
