# Pothole Eye — AI Road Hazard Detection System

**Real-time pothole detection, tracking, and automated government reporting powered by YOLOv8 + RT-DETR on GPU.**

> Deployed on RunPod A40 (48 GB VRAM) · CUDA FP16 · Flask + SocketIO · Mapbox GL JS · MediaMTX WebRTC

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [GPU Deployment (RunPod)](#gpu-deployment-runpod)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [REST API Reference](#rest-api-reference)
- [Configuration Reference](#configuration-reference)
- [Detection Pipeline](#detection-pipeline)
- [Severity Classification](#severity-classification)
- [Track State Machine](#track-state-machine)
- [Performance](#performance)

---

## Overview

Pothole Eye processes live dashcam footage or uploaded video to:

1. **Detect** potholes using a custom-trained YOLOv8m model (mAP50 = 0.729) with RT-DETR-L as an optional primary detector
2. **Segment** the road surface (YOLOv8l-seg → SegFormer → DeepLabV3 fallback chain) to eliminate false positives on sky, buildings, and vegetation
3. **Estimate depth** (Depth Anything V2 Metric → MiDaS → geometric fallback) for real-world severity scoring
4. **Track** detections across frames using ByteTrack with a 4-state FSM gating (TENTATIVE → CONFIRMED → MATURE → ARCHIVED)
5. **Deduplicate** spatially using 5m Haversine distance to avoid repeat DB entries for the same pothole
6. **Alert** the driver in real time with Web Audio tones and visual overlays
7. **Report** confirmed potholes to the Maryland SHA automatically by email with GPS, snapshot, and 3-tier escalation

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         POTHOLE EYE  v2                                  │
│                                                                          │
│  ┌─────────────────────┐    ┌────────────────────────────────────────┐  │
│  │    Input Sources     │    │         GPU Inference Worker           │  │
│  │                      │    │         (RunPod A40 · CUDA FP16)       │  │
│  │  Browser Camera ─────┼───▶│  1. YOLOv8m custom (primary)          │  │
│  │  GoPro / RTSP   ─────┼───▶│     └── RT-DETR-L (optional)          │  │
│  │  Video Upload   ─────┼───▶│                                        │  │
│  └─────────────────────┘    │  2. YOLOv8l-seg → SegFormer-B2         │  │
│                              │     → DeepLabV3 → geometric            │  │
│                              │     (road surface mask)                │  │
│                              │                                        │  │
│                              │  3. Depth Anything V2 Metric           │  │
│                              │     → MiDaS → geometric depth         │  │
│                              │                                        │  │
│                              │  4. PerceptionFusion                   │  │
│                              │     ├── Road mask gate (≥35% coverage) │  │
│                              │     ├── Metric depth extraction        │  │
│                              │     └── Severity scoring (L1–L4)       │  │
│                              └────────────────────────────────────────┘  │
│                                             │                            │
│                              ┌──────────────▼──────────────┐            │
│                              │     ByteTrack Tracker        │            │
│                              │  (two-stage IoU association) │            │
│                              └──────────────┬──────────────┘            │
│                                             │                            │
│                              ┌──────────────▼──────────────┐            │
│                              │   PotholeStateMachine        │            │
│                              │  TENTATIVE → CONFIRMED       │            │
│                              │           → MATURE           │            │
│                              │           → ARCHIVED         │            │
│                              └──────────────┬──────────────┘            │
│                                             │  CONFIRMED+ only          │
│         ┌───────────────────────────────────┼──────────────┐            │
│         ▼                                   ▼              ▼            │
│  ┌──────────────────┐    ┌──────────────────────┐  ┌─────────────────┐  │
│  │  Alert Manager   │    │  SQLite + Spatial DB  │  │  Mapbox GL JS   │  │
│  │  ├─ DANGER <5m   │    │  ├─ 5m Haversine dedup│  │  ├─ Clusters    │  │
│  │  ├─ CAUTION <15m │    │  ├─ DBSCAN clustering │  │  ├─ Heatmap     │  │
│  │  ├─ Web Audio    │    │  └─ Track lifecycle   │  │  └─ GPS path    │  │
│  │  └─ Frame border │    └──────────────────────┘  └─────────────────┘  │
│  └──────────────────┘                 │                                  │
│                              ┌────────▼────────┐                         │
│                              │  MDOT Reporting  │                         │
│                              │  3-tier email    │                         │
│                              └─────────────────┘                         │
│                                                                          │
│   Flask + SocketIO  ·  REST API (GeoJSON)  ·  MediaMTX WebRTC           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Implementation |
|---|---|
| Pothole detection | Custom YOLOv8m (mAP50=0.729) · RT-DETR-L optional · single-class |
| Road surface mask | YOLOv8l-seg → SegFormer-B2 → DeepLabV3 → geometric (fallback chain) |
| Lane detection | Canny + HoughLinesP road polygon, ANDed with DL segmentation mask |
| Metric depth | Depth Anything V2 Metric Outdoor → MiDaS → geometric (fallback chain) |
| Perception fusion | Road mask gate (35%) + depth extraction + L1–L4 severity scoring |
| Object tracking | ByteTrack two-stage high/low confidence IoU association |
| Track state machine | TENTATIVE → CONFIRMED (age≥2, EMA≥0.25) → MATURE (age≥10) |
| Severity levels | L1 COSMETIC / L2 MODERATE / L3 SEVERE / L4 CRITICAL |
| Real-time alerts | Web Audio API (880 Hz DANGER · 440 Hz CAUTION) + red frame border |
| Live streaming | MediaMTX RTSP → WebRTC WHEP (200–400 ms) via NVENC encoding |
| Browser camera | WebSocket base64 frames + browser Geolocation API |
| Video upload | Background Celery worker · real-time progress · annotated MP4 output |
| Map | Mapbox GL JS dark theme · clusters · heatmap · 3D · light/dark toggle |
| GPS geo-tagging | Locked at track CONFIRMATION — never at tentative stage |
| Spatial dedup | 5 m Haversine — upsert existing record within radius |
| Geo-clustering | DBSCAN for map-level density analysis |
| MDOT reporting | Automated email to SHA with snapshot, GPS, 3-tier escalation |
| Database | SQLAlchemy + SQLite · v2 schema migration on startup |
| REST API | 15 endpoints — CRUD, GeoJSON, GPS status, performance metrics |
| Docker GPU | `docker-compose.gpu.yml` with NVIDIA runtime for RunPod |

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Mapbox public token](https://account.mapbox.com/auth/signup/) (free tier works)
- Model weights (see §Download Weights below)

### 1. Clone and Install

```bash
git clone https://github.com/your-username/Pothole-I.git
cd Pothole-I
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Open .env and set at minimum:
#   MAPBOX_ACCESS_TOKEN=pk.xxx
#   POTHOLE_MODEL_PATH=models/pothole_yolov8.pt
```

### 3. Download Model Weights

Weights are not stored in git (too large). Download from HuggingFace:

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="mdmis001/pothole-eye-models",
    filename="pothole_yolov8.pt",
    local_dir="models/"
)
EOF
```

### 4. Download the Training Dataset (only needed to retrain)

The dataset is gitignored due to size (198 MB). Get it from Roboflow:

```
https://universe.roboflow.com/mahades-workspace/pothole-detection-sjbkl-dwm0o
```

Extract into `Pothole-Detection-1/` so `Pothole-Detection-1/data.yaml` exists.

---

## GPU Deployment (RunPod)

This project runs exclusively on **RunPod A40 GPU**. All inference uses `INFERENCE_DEVICE=cuda:0`.

### Step 1 — Configure deploy script

Open [scripts/deploy_runpod.sh](scripts/deploy_runpod.sh) and set your pod's connection details:

```bash
RUNPOD_HOST="your.runpod.ip.here"   # from RunPod dashboard → Connect → SSH over TCP
RUNPOD_PORT="your_port_here"
```

### Step 2 — Configure GPU environment

```bash
cp .env.gpu.example .env.gpu
# Fill in: HF_TOKEN, MAPBOX_ACCESS_TOKEN, SECRET_KEY
```

### Step 3 — Sync and launch

```bash
# Sync code to RunPod + pull back trained weights
bash scripts/deploy_runpod.sh

# SSH in and start
ssh -i ~/.ssh/id_ed25519 -p YOUR_PORT root@YOUR_HOST
cd /workspace/Pothole-I
mkdir -p .cache/torch .cache/huggingface logs
docker compose -f docker-compose.gpu.yml up --build -d
```

### Step 4 — Access the dashboard

```bash
# SSH tunnel
ssh -L 5001:localhost:5001 -i ~/.ssh/id_ed25519 -p YOUR_PORT root@YOUR_HOST
```

Open **http://localhost:5001**

### Verify GPU and models

```bash
docker exec ai-eyes-on-the-road-gpu python3 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
docker logs ai-eyes-on-the-road-gpu 2>&1 | grep -E 'model|FOUND|pothole'
```

---

## Training the Model

```bash
# Standard production run (~3–4 hrs on A40, mAP50 > 0.72)
python training/train_gpu.py

# Fast baseline (~25 min, mAP50 > 0.60)
python training/train_gpu.py --model yolov8n.pt --imgsz 640 --batch 64 --epochs 100

# Maximum quality (~8 hrs, mAP50 > 0.82)
python training/train_gpu.py --model yolov8l.pt --imgsz 1280 --batch 8 --epochs 200
```

**Default training config:** `yolov8m.pt · imgsz=1280 · batch=16 · epochs=150 · AMP FP16`

After training, `best.pt` is automatically copied to `models/pothole_yolov8.pt` and pushed to HuggingFace (`HF_MODEL_REPO` in `.env.gpu`).

### Evaluate a trained model

```bash
python training/evaluate.py \
  --weights models/pothole_yolov8.pt \
  --dataset-dir Pothole-Detection-1 \
  --split val --imgsz 1280
```

---

## Project Structure

```
Pothole-I/
├── app.py                       Flask app · SocketIO · camera/upload workers
├── config.py                    Central config (all env vars + directory init)
├── requirements.txt
│
├── pipeline/
│   ├── processor.py             PipelineProcessor — per-frame orchestrator
│   ├── gpu_worker.py            GPUInferenceWorker — batched GPU inference
│   ├── detection.py             PotholeDetector (YOLOv8m / RT-DETR-L)
│   ├── segmentation.py          RoadSegmenter (YOLOv8l-seg / SegFormer / DeepLabV3)
│   ├── lane_detection.py        LaneDetector (Canny + HoughLinesP polygon)
│   ├── depth.py                 DepthEstimator (Depth Anything V2 / MiDaS / geometric)
│   ├── fusion.py                PerceptionFusion (road gate + depth + severity)
│   ├── scheduler.py             AdaptiveFrameScheduler
│   └── ingestion.py             FramePacket dataclass
│
├── tracking/
│   ├── bytetrack_adapter.py     PotholeByteTracker (ultralytics BYTETracker)
│   ├── state_machine.py         PotholeStateMachine (TENTATIVE→CONFIRMED→MATURE)
│   ├── pothole_memory.py        PotholeMemory (ByteTrack + FSM + DB persistence)
│   └── sort_tracker.py          SORT fallback (Kalman + Hungarian)
│
├── models/
│   └── model_manager.py         Lazy-load + cache all ML models
│
├── database/
│   ├── models.py                SQLAlchemy ORM: Pothole, DetectionEvent, Report
│   ├── operations.py            PotholeDB (CRUD + spatial dedup + geo-clustering)
│   └── geo_clustering.py        DBSCAN geo-clustering
│
├── api/
│   └── routes.py                REST API blueprint (15 endpoints)
│
├── alerts/
│   └── alert_manager.py         AlertManager (throttled SocketIO dispatch)
│
├── reporting/
│   └── reporter.py              MDOTReporter (initial + follow-up + escalation)
│
├── services/
│   ├── rtsp_streamer.py         NVENCRTSPStreamer → MediaMTX
│   └── video_queue.py           Video upload job queue
│
├── training/
│   ├── train_gpu.py             A40-optimized YOLOv8 training (main entry point)
│   ├── train.py                 General YOLOv8 training (device-agnostic)
│   ├── train_rtdetr.py          RT-DETR fine-tuning (3-stage)
│   ├── evaluate.py              Model evaluation — mAP, PR curve, F1
│   ├── export.py                ONNX / TorchScript export
│   ├── model_store.py           HuggingFace push/pull for model persistence
│   ├── validate_dataset.py      Dataset integrity check
│   ├── hyperparam_tune.py       Hyperparameter search
│   ├── inference.py             Standalone inference script
│   └── setup.py                 Dataset setup and environment verification
│
├── scripts/
│   ├── deploy_runpod.sh         rsync to RunPod + pull trained weights back
│   ├── start_app.sh             Bare-metal startup (Redis → Celery → Flask)
│   ├── evaluate.py              RunPod-path evaluate wrapper
│   └── verify_gpu.py            GPU + CUDA environment check
│
├── static/
│   ├── css/dashboard.css
│   ├── js/dashboard.js          SocketIO client + UI controller
│   ├── js/map.js                Mapbox GL JS map controller
│   ├── js/camera.js             GPSManager + camera source selection
│   ├── js/webrtc_stream.js      WebRTC WHEP client
│   └── img/logo.svg
│
├── templates/
│   └── index.html               Single-page dashboard
│
├── Dockerfile                   CPU image (dev)
├── Dockerfile.gpu               GPU image (CUDA 12.4 + PyTorch 2.4)
├── docker-compose.yml           CPU compose
├── docker-compose.gpu.yml       GPU compose (RunPod)
│
├── .env.example                 Environment template — copy to .env
├── .env.gpu.example             GPU environment template — copy to .env.gpu
├── .gitignore
└── .dockerignore

# Gitignored (not in repo):
#   .env / .env.gpu              — real credentials
#   models/pothole_yolov8.pt     — 50 MB, download from HuggingFace
#   Pothole-Detection-1/         — 198 MB dataset, download from Roboflow
#   database/potholes.db         — runtime data
#   uploads/ outputs/ runs/      — runtime + training artifacts
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/potholes` | List potholes (`?limit=N&format=flat\|full`) |
| GET | `/api/potholes/<id>` | Single pothole detail |
| DELETE | `/api/potholes/<id>` | Delete record |
| PATCH | `/api/potholes/<id>/location` | Assign GPS to ungeotagged pothole |
| POST | `/api/potholes/<id>/report` | Submit initial MDOT report |
| POST | `/api/potholes/<id>/personal-report` | Send personal email alert |
| GET | `/api/potholes/<id>/report-preview` | Preview report text |
| POST | `/api/potholes/<id>/mark-crossed` | Mark pothole as crossed |
| GET | `/api/map-data` | GeoJSON FeatureCollection (Mapbox source) |
| GET | `/api/mapbox-token` | Serve Mapbox token securely from env |
| GET | `/api/stats` | Detection statistics (total, severity breakdown) |
| GET | `/api/alerts` | Recent alert history (last 20) |
| GET | `/api/gps-status` | GPS source, coordinates, camera mode |
| GET | `/api/performance` | CPU / memory / thread metrics |
| GET | `/api/health` | Liveness check |
| GET | `/api/jobs/<job_id>/progress` | Video upload job progress |

---

## Configuration Reference

All settings are read from environment variables (`.env` for dev, `.env.gpu` for RunPod).

| Variable | Default | Description |
|---|---|---|
| `DETECTION_BACKEND` | `yolov8` | `yolov8` or `rtdetr` |
| `POTHOLE_MODEL_PATH` | `models/pothole_yolov8.pt` | Custom-trained single-class model |
| `YOLO_MODEL` | `yolov8m.pt` | General YOLO base model |
| `SEGMENTATION_BACKEND` | `yolov8seg` | `yolov8seg` \| `segformer` \| `deeplabv3` \| `geometric` |
| `DEPTH_BACKEND` | `depthanything` | `depthanything` \| `midas` \| `geometric` |
| `DETECTION_CONFIDENCE` | `0.20` | Detection confidence threshold |
| `POTHOLE_CONFIDENCE` | `0.20` | Custom model confidence threshold |
| `NMS_THRESHOLD` | `0.45` | Non-maximum suppression IoU |
| `POTHOLE_MIN_AREA` | `800` | Minimum bounding box area (px²) |
| `BYTETRACK_HIGH_THRESH` | `0.45` | ByteTrack stage-1 confidence |
| `BYTETRACK_MATCH_THRESH` | `0.80` | IoU matching threshold |
| `FUSION_ROAD_RATIO_MIN` | `0.35` | Min road mask overlap to accept detection |
| `SPATIAL_DEDUP_THRESHOLD_M` | `5.0` | Haversine deduplication radius (metres) |
| `ALERT_COOLDOWN_SEC` | `3.0` | Minimum seconds between alerts |
| `FRAME_SKIP` | `1` | Process every Nth frame |
| `MAX_FRAME_WIDTH` | `960` | Resize width before inference |
| `RTSP_ENABLED` | `True` | Enable MediaMTX RTSP streaming |
| `MAPBOX_ACCESS_TOKEN` | — | Required for map (get at mapbox.com) |
| `SMTP_USERNAME` | — | Gmail address for MDOT email reporting |
| `SMTP_PASSWORD` | — | Gmail app password |
| `HF_TOKEN` | — | HuggingFace write token for model persistence |
| `HF_MODEL_REPO` | — | HuggingFace repo for model weights |
| `INFERENCE_DEVICE` | `cuda:0` | PyTorch device (GPU only) |

---

## Detection Pipeline

Each frame goes through this sequence:

```
Frame
  │
  ├─ Segmentation (YOLOv8l-seg / SegFormer / DeepLabV3)  → DL road mask
  ├─ Lane Detection (Canny + HoughLinesP)                 → polygon mask
  │                        └─ _combine_masks() ──────────→ final road_mask
  │
  ├─ apply_mask(frame, road_mask)  → masked_frame (non-road pixels = black)
  │
  ├─ YOLOv8m / RT-DETR on masked_frame  → raw detections
  ├─ Anomaly CV detection on road_mask  → additional candidates
  │
  ├─ PerceptionFusion
  │    ├─ Road mask gate: bbox must overlap road ≥ 35%
  │    ├─ Depth Anything V2 on original frame → depth_map
  │    └─ Severity score (depth δ × 0.40 + area × 0.35 + shape × 0.15 + conf × 0.10)
  │
  ├─ ByteTrack → PotholeStateMachine → CONFIRMED tracks
  │
  ├─ Spatial dedup (5m Haversine) → upsert SQLite
  │
  └─ Annotate + HUD → SocketIO emit → browser
```

---

## Severity Classification

| Score | Level | Colour | Meaning |
|---|---|---|---|
| < 0.25 | L1 COSMETIC | Cyan | Surface cracking, no structural threat |
| 0.25 – 0.50 | L2 MODERATE | Orange | Noticeable hazard, report recommended |
| 0.50 – 0.75 | L3 SEVERE | Deep orange | Immediate report required |
| ≥ 0.75 | L4 CRITICAL | Red | Urgent — structural failure risk |

Scoring formula:
```
score = 0.40 × depth_delta   (depth std inside bbox → crater depth proxy)
      + 0.35 × area_score    (real-world area via pinhole projection)
      + 0.15 × shape_score   (mask circularity — irregular = structurally worse)
      + 0.10 × confidence    (detector certainty)
```

---

## Track State Machine

Single-frame detections never reach the database. Tracks must earn confirmation:

```
TENTATIVE ──(age≥2, EMA≥0.25)──▶ CONFIRMED ──(age≥10)──▶ MATURE
    │                                 │                       │
    └──(missed>5)──▶ LOST             └──(missed>8)──▶ LOST  └──(missed>30)──▶ ARCHIVED
```

- **TENTATIVE** — seen at least once, not yet trusted
- **CONFIRMED** — multi-frame evidence, written to DB, triggers alerts and GPS stamp
- **MATURE** — stable long-term track, lower alert frequency
- **ARCHIVED** — track ended, record preserved in DB

---

## Performance

**RunPod A40 (48 GB VRAM), CUDA FP16:**

| Stage | Latency |
|---|---|
| YOLOv8m detection | ~12 ms/frame |
| YOLOv8l-seg segmentation | ~22 ms/frame |
| Depth Anything V2 | ~35 ms/frame |
| Full pipeline (all models active) | < 80 ms/frame (~13 FPS) |
| SocketIO frame delivery | < 50 ms end-to-end |
| WebRTC live stream latency | 200–400 ms |

**Model metrics (validation split):**

| Metric | Value |
|---|---|
| mAP50 | 0.729 |
| Precision | 0.742 |
| Recall | 0.676 |
| Training config | YOLOv8m · imgsz=1280 · 150 epochs · A40 AMP FP16 |

---

## What's New in v2

| Component | v1 | v2 |
|---|---|---|
| Primary detector | YOLOv8n (COCO) | Custom **YOLOv8m** (nc=1, mAP50=0.729) |
| Optional detector | — | RT-DETR-L (transformer, anchor-free) |
| Tracker | SORT (Kalman + Hungarian) | **ByteTrack** (two-stage, occlusion-resilient) |
| Track gating | Simple frame count | **PotholeStateMachine** (4-state FSM with EMA) |
| Depth | MiDaS relative | **Depth Anything V2 Metric** (real metres, no calibration needed) |
| Segmentation | DeepLabV3 only | **YOLOv8l-seg → SegFormer → DeepLabV3** fallback chain |
| Lane masking | None | **Canny + HoughLinesP** polygon ANDed with DL mask |
| Severity | None | **L1–L4 scoring** (depth + area + shape + confidence) |
| Map | Google Maps | **Mapbox GL JS** (dark theme, clusters, heatmap, 3D) |
| Streaming | WebSocket frames | **MediaMTX WebRTC WHEP** (200–400 ms) |
| Deployment | CPU Docker | **RunPod A40 CUDA FP16** |
| Dedup radius | 10 m | **5 m Haversine** + DBSCAN geo-clustering |

---

## License

MIT — Research and educational use.

---

*Built with YOLOv8 · RT-DETR · Depth Anything V2 · ByteTrack · SegFormer · Flask · Mapbox GL JS · MediaMTX*
