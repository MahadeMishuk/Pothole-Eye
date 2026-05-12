<p align="center">
  <img src="Images/logo.png" width="150" alt="Logo">
</p>

# Pothole Eye — Real-time pothole detection and alerting system using deep learning

**Real-time pothole detection, tracking, and automated reporting powered by YOLOv8 + RT-DETR on GPU.**

> Deployed on RunPod GPUs · CUDA FP16 · Flask + SocketIO · Mapbox GL JS · MediaMTX WebRTC

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



---


---

*Built with YOLOv8 · RT-DETR · Depth Anything V2 · ByteTrack · SegFormer · Flask · Mapbox GL JS · MediaMTX*
