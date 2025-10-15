# Technical Documentation: Advanced Intrusion Detection System (In-Depth)

This document provides an in-depth, engineering-focused description of the system’s architecture, algorithms, module APIs, configuration surface, data formats, performance characteristics, and extension points. It reflects the current implementation under `intrusion_detection/` and associated CLI tools.

## 1) Architecture and Data Flow

The system is modular. A central processing loop orchestrates motion analysis, optional person filtering (YOLOv8), ROI-based damage detection, on-frame annotation, and concise incident logging.

```
[Video Reader] → [Preprocess] → [Background Subtraction] → [Morphology → Contours]
                                         ↓
                                 [YOLOv8 Person Detector]
                                         ↓
                              [Person-Filtered Motion Events]
                                         ↓
                           [ROI-based Heuristic Damage Detector]
                                         ↓
                [Annotation]  +  [Incident Recorder → JSON Report]
```

Core modules and roles:
- `video.py`: Video capture wrappers and utility helpers (OpenCV/imageio).
- `background.py`: Background subtractor setup and mask preparation.
- `motion.py`: Motion detection + person filtering + rendering (APIs detailed below).
- `damage.py`: Heuristic structural/damage detection over ROIs (doors/windows) with persistence/cooldown.
- `incidents.py`: Concise incident logging with motion period grouping and consolidated reporting.
- `roi_selector.py` and `select_rois.py`: Interactive ROI definition and storage in JSON.
- `processing.py`: Main pipeline integration (frame loop, detectors, recorder, IO).
- `config.py`: All configuration; also initializes logging and (optionally) the YOLO detector.
- `debug_damage_detection.py`: Live parameter tuning for the damage module.

## 2) Algorithms

### 2.1 Motion Detection with Person Filtering

Background subtraction uses OpenCV’s KNN subtractor to produce a binary foreground mask $M_t$ from a frame $I_t$. Mask refinement uses morphology (open/close + dilation). Contours with area below a threshold are discarded.

- Minimum motion area criterion:
  $A(C) > \rho_{area}·(H·W)$ where $\rho_{area} = \texttt{Config.MIN_CONTOUR_AREA_PERCENT}$.

If person detection is enabled, YOLOv8 runs on (optionally enhanced) frames. Person boxes are filtered by:
- Confidence: $c \ge \texttt{PERSON_CONFIDENCE_THRESHOLD}$
- Area bounds: $\texttt{MIN_PERSON_AREA} \le w·h \le \texttt{MAX_PERSON_AREA}$
- Aspect ratio: $\texttt{PERSON_ASPECT_RATIO_MIN} \le w/h \le \texttt{PERSON_ASPECT_RATIO_MAX}$

A motion contour is kept iff its IoU with any person box is at least `MIN_OVERLAP_RATIO`:

$\mathrm{IoU}(B_m, B_p) = \frac{|B_m \cap B_p|}{|B_m \cup B_p|} \ge \tau_{iou}$

If `PERSON_ONLY_MODE=True` and no person is detected, all motion is suppressed. As a fallback, if YOLO detects persons but overlap is low, synthetic contours are generated from person boxes to preserve tracking.

Temporal debouncing: a persistence counter smooths binary motion decisions over `DETECTION_PERSISTENCE` frames to reduce flicker.

Contrast and gamma enhancements are optionally applied before YOLO to aid detection in low-light scenes (CLAHE + gamma LUT).

### 2.2 Heuristic Damage Detection (Doors/Windows)

Per ROI, the detector keeps a grayscale baseline and an exponential average of edge activity. For each frame, three signals are computed:

1) Mean absolute difference to baseline (intensity change):

$D_t = \frac{1}{|\Omega|}\sum_{(x,y)\in\Omega} |I_t(x,y) - B(x,y)|$

2) Edge spike ratio: apply Canny to current ROI; let $E_t$ be edge pixel count and $\bar E$ an EMA of edge counts. An edge spike holds if $E_t/(\bar E + \epsilon) \ge \texttt{DAMAGE_EDGE_SPIKE_RATIO}$.

3) Fragmentation: number of small contours (area in [5, 150]) in the edge map (proxy for shards/splinters): $F_t$.

Event logic:
- Forced Entry (door): $D_t \ge \texttt{DAMAGE_DIFF_THRESHOLD}$ and edge spike.
- Shatter (window): $F_t \ge \texttt{DAMAGE_FRAGMENT_COUNT_THRESHOLD}$ and edge spike.

An internal per-ROI persistence counter requires events to sustain for `DAMAGE_MIN_PERSISTENCE` frames. A cooldown of `DAMAGE_EVENT_COOLDOWN` frames prevents immediate retriggers. Confirmed events are retained for on-screen visibility over `DAMAGE_EVENT_HOLD_FRAMES`.

### 2.3 Concise Incident Logging

Instead of logging every frame, `IncidentRecorder` groups motion frames into contiguous “intrusion periods” with a maximum allowed gap (≤5 frames). The final report contains:
- A summary (video, total frames, counts, output video, key configuration snapshot).
- `motion_summary` with [start_frame, end_frame, duration_frames, max_persons_detected].
- Full list of damage incidents (timestamp, ROI, event type, score, ROI coordinates).
- Basic statistics (processing duration, rates, first/last frames).

A consolidated report across multiple videos can be generated via `generate_summary_report`.

## 3) Public APIs and Key Classes

### 3.1 `motion.py`

- `@dataclass Box(x, y, w, h, conf)`
  - `to_tuple() -> (x, y, w, h)`

- `class MotionDetector(cfg: Config, person_detector=None)`
  - `detect_persons_in_frame(frame) -> List[Box]`
    - Runs YOLOv8 (if enabled), applies confidence/area/aspect-ratio filters.
  - `filter_motion_by_persons(contours, person_boxes) -> List[np.ndarray]`
    - Keeps contours overlapping a person by ≥ `MIN_OVERLAP_RATIO`.
  - `detect(frame, fg_mask) -> (motion_detected, contours, bbox, person_boxes)`
    - Returns debounced motion boolean, refined contours, aggregate bbox, and person boxes.
  - `draw_motion(frame, contours, bbox, person_boxes) -> frame`
    - Renders contours, bboxes, and person boxes.

Notes:
- Input `fg_mask` is the refined foreground from background subtraction.
- Internal CLAHE/gamma enhancement for person detection controlled via `Config`.

### 3.2 `damage.py`

- `@dataclass DamageEvent(roi_type, roi_index, event_type, frame_index, score)`

- `class DamageDetector(cfg: Config)`
  - State per ROI: `baseline`, `edge_avg` (EMA), `persist` (frames), `last_event_frame`.
  - `process_frame(frame) -> List[DamageEvent]`
    - Evaluates all ROIs (doors then windows), returns confirmed events; maintains `recent_events` for hold-drawing.
  - `annotate(frame, events, cfg, recent_events=None) -> frame`
    - Draws ROI rectangles with labels and keeps recent ones for visibility.

Event semantics:
- `event_type='forced_entry'` for door triggers; `score=diff_mean`.
- `event_type='shatter'` for window triggers; `score=fragment_count`.

### 3.3 `incidents.py`

- `@dataclass MotionIncident(timestamp, frame_number, video_file, event_type, person_count, bounding_boxes, confidence)`
- `@dataclass DamageIncident(timestamp, frame_number, video_file, roi_type, roi_index, event_type, score, roi_coordinates)`
- `@dataclass VideoProcessingSummary(video_file, start_time, end_time, total_frames, motion_incidents, damage_incidents, output_video, configuration)`

- `class IncidentRecorder(cfg: Config, video_file: str)`
  - `log_motion_event(frame_number, event_type='motion_detected', person_count=0, bounding_boxes=None, confidence=None)`
  - `log_damage_event(frame_number, event: DamageEvent)`
  - `finalize(output_video_path, total_frames) -> str`
    - Writes concise JSON report under `Config.INCIDENT_LOG_DIR`.
  - `_create_motion_summary() -> dict`
- `generate_summary_report(cfg, incident_files: List[str]) -> str`

## 4) Configuration Reference (`config.py`)

Below are primary configuration attributes with representative defaults. See `intrusion_detection/config.py` for the authoritative source.

- Paths
  - `INPUT_VIDEO_DIR='./input_videos/'`
  - `OUTPUT_VIDEO_DIR='./output_videos/'`
  - `DATA_DIR='./data/'`
  - `SAVE_MOTION_DIR='./data/detected_motion/'`
  - `INCIDENT_LOG_DIR='./data/incidents/'`

- Video / Processing
  - `FRAME_WIDTH=640`, `FRAME_HEIGHT=480`
  - `PROCESS_EVERY_N_FRAMES=3` (frame skipping)

- Motion Detection
  - `MIN_CONTOUR_AREA_PERCENT=0.015`
  - `DETECTION_PERSISTENCE=3`

- Person Detection (YOLO)
  - `ENABLE_PERSON_DETECTION=True`
  - `PERSON_ONLY_MODE=True`
  - `PERSON_CONFIDENCE_THRESHOLD=0.5`
  - `MIN_PERSON_AREA=1000`, `MAX_PERSON_AREA=200000`
  - `PERSON_ASPECT_RATIO_MIN=0.3`, `PERSON_ASPECT_RATIO_MAX=4.0`
  - `MIN_OVERLAP_RATIO=0.1`

- Enhancement (for YOLO input)
  - `AUTO_CONVERT_GRAYSCALE=True`
  - `ENHANCE_CONTRAST=True`, `CLAHE_CLIP_LIMIT=1.5`, `CLAHE_TILE_SIZE=8`
  - `APPLY_GAMMA_CORRECTION=True`, `GAMMA=1.2`, `APPLY_GAMMA_TO_OUTPUT=True`

- Pre-blur / Denoise
  - `DENOISE_BEFORE_BG=True`, `DENOISE_KERNEL_SIZE=3`, `DENOISE_SIGMA=0`

- Damage Detection
  - `ENABLE_DAMAGE_DETECTION=True`
  - `DOOR_ROIS=[...]`, `WINDOW_ROIS=[...]` (set via ROI tools)
  - `DAMAGE_DIFF_THRESHOLD=40.0`
  - `DAMAGE_EDGE_SPIKE_RATIO=0.5`
  - `DAMAGE_FRAGMENT_COUNT_THRESHOLD=4`
  - `DAMAGE_MIN_PERSISTENCE=2`
  - `DAMAGE_EVENT_COOLDOWN=300` (frames)
  - `DAMAGE_EVENT_HOLD_FRAMES=60`

- Background Subtractor
  - `BACKGROUND_HISTORY=500`, `BACKGROUND_THRESHOLD=400`

- Recording / Alerts / Display
  - `SAVE_DETECTED_MOTION=True`, `SAVE_ALERT_VIDEO=True`, `VIDEO_CLIP_DURATION=10`
  - `ALERT_COOLDOWN=30`
  - `SHOW_PROCESSED_FRAMES=True`, `DASHBOARD_UPDATE_INTERVAL=5`

- Logging
  - `LOG_FILE='motion_detection_log.csv'`
  - `LOG_LEVEL=logging.INFO`

- Incident Logging
  - `ENABLE_INCIDENT_LOGGING=True`
  - `INCIDENT_LOG_FORMAT='json'`
  - `LOG_MOTION_EVENTS=True`, `LOG_DAMAGE_EVENTS=True`

Utilities:
- `setup_directories(cfg) -> list[str]`
- `setup_logging(cfg) -> logging.Logger`
- `init_yolo(cfg)` → YOLO detector or `None`
- `get_output_video_path(input_video_name, suffix='enhanced')`

## 5) Data Formats

### 5.1 ROI JSON Format (`data/rois/{video}_rois.json`)
A typical schema is:
```json
{
  "video": "intruder_4.mp4",
  "rois": [
    { "label": "door", "x": 320, "y": 180, "w": 120, "h": 240 },
    { "label": "window", "x": 120, "y": 100, "w": 160, "h": 120 }
  ]
}
```
ROIs are interpreted in the resized frame coordinate system defined by `FRAME_WIDTH/FRAME_HEIGHT`.

### 5.2 Incident Report (Concise JSON)
Generated per processed video under `data/incidents/`:
```json
{
  "summary": {
    "video_file": "intruder_4.mp4",
    "start_time": "...",
    "end_time": "...",
    "total_frames": 1787,
    "motion_incidents": 163,
    "damage_incidents": 6,
    "output_video": "intruder_4_enhanced_20251015_012121.mp4",
    "configuration": { "person_only_mode": true, ... }
  },
  "motion_summary": {
    "total_intrusion_periods": 3,
    "total_motion_frames": 163,
    "max_consecutive_frames": 65,
    "intrusion_periods": [
      { "start_frame": 385, "end_frame": 449, "duration_frames": 65, "max_persons_detected": 1 }
    ]
  },
  "damage_incidents": [
    {
      "timestamp": "...",
      "frame_number": 1234,
      "video_file": "intruder_4.mp4",
      "roi_type": "window",
      "roi_index": 0,
      "event_type": "shatter",
      "score": 7.0,
      "roi_coordinates": [100, 150, 50, 50]
    }
  ],
  "statistics": { "processing_duration_seconds": 4.21, ... }
}
```

A consolidated report file can also be produced from multiple incident logs.

## 6) Performance and Complexity

- Per-frame cost: background subtraction $\mathcal{O}(HW)$; morphology and contour detection $\approx \mathcal{O}(HW)$.
- YOLOv8 cost depends on resolution/model; `yolov8n` balances speed and accuracy.
- Throughput can be tuned via:
  - `FRAME_WIDTH/FRAME_HEIGHT` (downscale)
  - `PROCESS_EVERY_N_FRAMES` (frame skipping)
  - Restrict YOLO application (e.g., detect intermittently or within motion bbox)
- Memory: dominated by frame buffers and detector; incident reports are concise (KBs).

## 7) Error Handling, Logging, and Diagnostics

- Logging: configured in `setup_logging`, default file `motion_detection_system.log` + console.
- YOLO detection failures are caught with `try/except` and logged at debug level; pipeline continues.
- ROI index bounds are checked before annotation/logging to avoid crashes on malformed ROI files.
- The debug tool (`debug_damage_detection.py`) enables live threshold tuning for damage detection to mitigate false alarms or misses.

## 8) Extension Points

- Swap or augment YOLO model by changing `init_yolo` to different weights/sizes.
- Replace background subtractor in `background.py` (e.g., MOG2) or introduce stabilization.
- Add new ROI labels and custom event heuristics in `damage.py`.
- Extend `IncidentRecorder` to write CSV/Parquet or push to a database/REST endpoint.
- Add alert transports in `alerts.py` (email/Telegram/webhook) observing `ALERT_COOLDOWN`.

## 9) Security and Privacy Considerations

- The system does not perform face recognition; person detection is class-level only.
- Ensure stored videos/reports are access-controlled. Consider encrypting archives at rest.
- Clearly communicate monitoring to comply with applicable laws and policies.

## 10) Known Limitations and Mitigations

- Sensitivity to camera motion: consider stabilization or masking unstable regions.
- Environment-specific tuning: thresholds may require adjustment via the debug tool.
- Low-light performance: enable CLAHE and gamma; consider IR-capable cameras.
- Computational load: use `yolov8n`, downscale, and frame skipping; or offload detection to intervals.

## 11) Reproducibility Checklist

- Pin environment via `requirements.txt`.
- Save `config.py` snapshot with reports for exact parameters.
- Keep ROI files under `data/rois/` versioned alongside videos for consistent evaluation.

---
For domain context and publication-ready narrative, see `JOURNAL_MANUSCRIPT.md`. This document focuses on the technical and API aspects used by integrators and developers.
