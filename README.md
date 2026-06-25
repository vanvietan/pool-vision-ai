# Pool Vision AI — MVP

Computer-vision "chess engine for pool". Upload a photo of a pool table; the
system detects the table, balls and pockets, then suggests the best shot and
draws it as an overlay.

This MVP is a **thin vertical slice**: image upload → table detect → ball
detect (OpenCV) → ghost-ball shot engine → overlay → browser display.
No DB, video, AR, or spin yet (see Roadmap in the PRD).

## Architecture

```
Browser (Vite/React/TS :5173)
   │  POST /analyze (multipart image)
   ▼
Go / Gin proxy (:8090)
   │  forwards multipart
   ▼
Python / FastAPI + OpenCV (:8000)   ← all CV + geometry
   ▲  JSON { balls, pockets, shot, overlay_png(base64) }
```

```
ai-service/   Python FastAPI + OpenCV  — table, detect, geometry, overlay
backend/      Go + Gin                 — thin proxy (DB/auth land here later)
frontend/     Vite + React + TS        — upload + overlay display
```

Detection sits behind a `Detector` interface (`ai-service/app/detect.py`).
`make_detector()` picks the backend from `DETECTOR` env (`opencv` default,
`yolo` opt-in) — no pipeline changes either way.

### YOLO backend (opt-in, higher accuracy)

```bash
cd ai-service
.venv/bin/pip install -r requirements-yolo.txt   # heavy: pulls torch
DETECTOR=yolo .venv/bin/uvicorn app.main:app --port 8000
```

Defaults to COCO `sports ball` (stock `yolov8n.pt`, auto-downloaded). For the
PRD >95% target, train/point a pool-specific model and set its ball class ids:

```bash
DETECTOR=yolo YOLO_MODEL=pool.pt YOLO_CLASSES=0,1 ... uvicorn ...
```

| Env            | Default        | Meaning                          |
| -------------- | -------------- | -------------------------------- |
| `DETECTOR`     | `opencv`       | `opencv` \| `yolo`               |
| `YOLO_MODEL`   | `yolov8n.pt`   | model path/name                  |
| `YOLO_CONF`    | `0.25`         | confidence threshold             |
| `YOLO_CLASSES` | `32`           | comma-sep class ids to keep      |

If ultralytics/torch or the model is missing, it logs a warning and falls
back to `OpenCVDetector` so the pipeline still runs.

## Run — one command (recommended)

```bash
./dev.sh        # or: make dev
```

Starts all 3 services. First run auto-creates the Python venv, installs
`requirements.txt` and `npm install` (later runs skip setup). **Ctrl-C stops
everything.** Then open **http://localhost:5173** and upload a pool-table photo.

| Service  | URL                     |
| -------- | ----------------------- |
| UI       | http://localhost:5173   |
| backend  | http://localhost:8090   |
| ai       | http://localhost:8000   |

Override ports/targets with env vars: `AI_PORT`, `PORT` (backend),
`FRONTEND_PORT`, `AI_SERVICE_URL`, `VITE_API_BASE`.

> Backend uses **8090** (not the Go default 8080) because 8080 was already
> taken locally.

Prereqs: Python 3, Go, Node.js installed.

## Run — manual (3 terminals)

<details><summary>expand</summary>

```bash
# 1. AI service (:8000)
cd ai-service && python3 -m venv .venv \
  && .venv/bin/pip install -r requirements.txt \
  && .venv/bin/uvicorn app.main:app --port 8000

# 2. Backend proxy (:8090)
cd backend && PORT=8090 go run .

# 3. Frontend (:5173)
cd frontend && npm install && npm run dev
```
</details>

## Manual table corners (oblique / real photos)

Classic-CV auto corner-finding is unreliable on steep, table-height photos.
The UI lets you click the 4 table corners (TL → TR → BR → BL); click 3 and the
4th is guessed by parallelogram completion, then drag any corner — including
off the photo edge for occluded/out-of-frame corners — to refine. With corners
supplied the server warps from them directly (no auto-detect).

`/analyze` accepts an optional `corners` form field: JSON `[[x,y],...]` of 4
points, TL,TR,BR,BL, normalized to `[0,1]` (values may fall outside for
off-frame corners). Omit it to auto-detect.

```bash
curl -F image=@photo.heic \
     -F 'corners=[[0.1,0.4],[0.95,0.38],[0.95,0.72],[0.02,0.78]]' \
     http://localhost:8090/analyze
```

> HEIC/HEIF (iPhone) photos are decoded via pillow-heif; EXIF orientation is
> applied so sideways photos sit upright.

## Quick test without the UI

```bash
# generate a synthetic table image (already produces ai-service/sample.jpg in dev)
curl -F image=@ai-service/sample.jpg http://localhost:8000/analyze   # direct
curl -F image=@ai-service/sample.jpg http://localhost:8090/analyze   # via Go
```

Response shape:
```json
{
  "table_detected": true,
  "balls": [{ "id": 0, "x": 599, "y": 250, "radius": 12, "is_cue": false }],
  "pockets": [{ "name": "top_left", "x": 0, "y": 0 }],
  "shot": { "target_ball": 1, "target_pocket": "top_right",
            "difficulty": 0.31, "success_rate": 0.33 },
  "overlay_png": "<base64 PNG>"
}
```

## How the shot engine works

Ghost-ball method (`ai-service/app/geometry.py`): for every
(object ball, pocket) pair it computes the aim point where the cue ball must
arrive (`ghost = object − 2R·unit(pocket−object)`), the cut angle, rejects
shots blocked by other balls or aimed the wrong way, then scores difficulty
from cut angle + travel distance. Lowest-difficulty unblocked shot wins.

## Known gaps / next

- OpenCV Hough detection is brittle to lighting/cloth color. YOLO backend
  now wired via the `Detector` seam (`DETECTOR=yolo`); next step is a
  pool-specific trained model to hit the PRD >95% target.
- No persistence → add Postgres + training history (PRD Phase 1).
- Add pytest for `geometry.py` (pure functions, high value).
- Tune `LOWER_GREEN`/`UPPER_GREEN` in `table.py` and Hough params in
  `detect.py` for real photos.
```
