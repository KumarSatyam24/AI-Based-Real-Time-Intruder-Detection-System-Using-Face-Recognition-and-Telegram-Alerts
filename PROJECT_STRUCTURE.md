# Project Structure

```
facial_detection_system/
│
├── intrusion_detection/          # Core Python package
│   ├── __init__.py
│   ├── config.py                 # Configuration and settings
│   ├── video.py                  # Video capture wrapper
│   ├── background.py             # Background subtraction
│   ├── motion.py                 # Motion detection + YOLO
│   ├── damage.py                 # Damage detection
│   ├── incidents.py              # Incident logging
│   ├── roi_selector.py           # ROI management
│   ├── processing.py             # Video processing pipeline
│   ├── alerts.py                 # Alert system
│   ├── system.py                 # Batch orchestration
│   └── interactive.py            # Interactive utilities
│
├── run_intrusion_detection.py    # Main CLI runner
├── select_rois.py                # ROI selection tool
├── debug_damage_detection.py     # Debug visualization
├── view_incidents.py             # Report viewer
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── README.md                     # Documentation
├── .gitignore                    # Git ignore rules
│
├── input_videos/                 # Place videos here
│   └── .gitkeep
│
├── output_videos/                # Enhanced videos output
│   └── .gitkeep
│
└── data/                         # Data files
    ├── incidents/                # JSON incident reports
    │   └── .gitkeep
    ├── rois/                     # ROI definitions
    │   └── .gitkeep
    └── detected_motion/          # Motion clips (optional)
        └── .gitkeep
```

## Key Files

### Scripts
- **run_intrusion_detection.py** - Main entry point for processing videos
- **select_rois.py** - Interactive tool to define regions of interest
- **debug_damage_detection.py** - Real-time threshold tuning
- **view_incidents.py** - View and analyze incident reports

### Core Modules
- **config.py** - All configuration parameters and thresholds
- **motion.py** - Motion detection and person tracking logic
- **damage.py** - Damage detection algorithms
- **incidents.py** - Incident logging and report generation
- **processing.py** - Main video processing pipeline

### Configuration
- **requirements.txt** - Python package dependencies
- **.gitignore** - Files to exclude from version control
- **LICENSE** - MIT License terms

## Directory Purpose

- **input_videos/** - Place your video files here for processing
- **output_videos/** - Enhanced videos with detections are saved here
- **data/incidents/** - JSON incident reports for each video
- **data/rois/** - Saved ROI definitions per video
- **data/detected_motion/** - Optional motion clip storage
