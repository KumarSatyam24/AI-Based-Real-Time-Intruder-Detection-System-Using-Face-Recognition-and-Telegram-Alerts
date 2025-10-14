# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Process Videos

```bash
# All videos in input_videos/
python run_intrusion_detection.py --all

# Specific video
python run_intrusion_detection.py --select video.mp4

# Without YOLO person detection
python run_intrusion_detection.py --all --disable-yolo

# Limit frames (for testing)
python run_intrusion_detection.py --all --max-frames 300
```

### View Reports

```bash
# Latest report
python view_incidents.py

# Specific report
python view_incidents.py --file data/incidents/report.json

# List all reports
python view_incidents.py --list

# Show motion details
python view_incidents.py --show-motion
```

### Define ROIs

```bash
python select_rois.py input_videos/video.mp4
```

**Controls:**
- Click & drag to draw rectangle
- `D` - Mark as door
- `W` - Mark as window
- `U` - Undo last ROI
- `S` - Save and exit

### Debug Thresholds

```bash
python debug_damage_detection.py input_videos/video.mp4
```

**Controls:**
- `SPACE` - Pause/Resume
- `R` - Reset detector
- `Q` - Quit
- Trackbars - Adjust thresholds

## Configuration

Edit `intrusion_detection/config.py`:

### Motion Detection
```python
MIN_CONTOUR_AREA_PERCENT = 0.015
DETECTION_PERSISTENCE = 3
```

### Person Detection
```python
PERSON_ONLY_MODE = True
PERSON_CONFIDENCE_THRESHOLD = 0.5
MIN_PERSON_AREA = 1000
```

### Damage Detection
```python
DAMAGE_DIFF_THRESHOLD = 40.0
DAMAGE_EDGE_SPIKE_RATIO = 0.5
DAMAGE_FRAGMENT_COUNT_THRESHOLD = 4
DAMAGE_MIN_PERSISTENCE = 2
DAMAGE_EVENT_HOLD_FRAMES = 60
```

### Enhancement
```python
CLAHE_CLIP_LIMIT = 1.5
GAMMA = 1.2
```

## Output

- **Videos**: `output_videos/video_enhanced_TIMESTAMP.mp4`
- **Reports**: `data/incidents/video_incidents_TIMESTAMP.json`
- **ROIs**: `data/rois/video_rois.json`

## Common Issues

**No output videos:**
- Check `motion_detection_system.log`
- Verify videos in `input_videos/`

**YOLO not working:**
- First run downloads model automatically
- Or use `--disable-yolo`

**Low accuracy:**
- Use `debug_damage_detection.py` to tune
- Adjust thresholds in `config.py`

**False positives:**
- Increase `DAMAGE_MIN_PERSISTENCE`
- Raise `PERSON_CONFIDENCE_THRESHOLD`

## Tips

- ROIs are auto-loaded per video
- Reports use concise format (5KB vs 52KB)
- Damage highlights persist for 60 frames
- Person-only mode reduces false positives
