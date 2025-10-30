# Advanced Intrusion Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-Ready-purple.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</p>

<p align="center">
  A comprehensive, real-time intrusion detection system featuring motion detection, person tracking, damage detection (forced entry, window shattering), and automatic incident reporting. Built with OpenCV, YOLO, and modern computer vision techniques.
</p>

<!-- 
<p align="center">
  <img src="docs/demo.gif" width="80%">
</p>
-->

## 🌟 Key Features

- ✅ **Motion Detection**: High-performance background subtraction (KNN) with morphological filtering to detect movement in restricted zones.
- 👤 **Person Tracking**: Integrates YOLOv8 to specifically identify and track humans, significantly reducing false positives from other movements.
- 🚪 **Damage Detection**: A multi-heuristic algorithm to detect signs of forced entry, such as window shattering or door impacts, by analyzing frame differences, edge spikes, and fragmentation.
- � **Telegram Notifications**: Real-time mobile alerts with frame images sent directly to your phone when motion, persons, or damage is detected. *(NEW!)*
- �📊 **Concise Incident Reporting**: Automatically generates detailed JSON reports that summarize events into "intrusion periods," reducing log size by over 90% compared to frame-by-frame logging.
- 🎯 **Interactive ROI Management**: Easy-to-use tool to draw and define specific Regions of Interest (e.g., doors, windows) for targeted monitoring.
- 🔧 **Real-time Debugging**: Includes a debug tool with interactive trackbars to fine-tune damage detection sensitivity live on a video stream.
- 📹 **Batch Processing**: Capable of processing multiple video files in a single run, generating individual reports for each.
- ⚙️ **Highly Configurable**: A central `config.py` file allows for easy tuning of all detection parameters and system settings.

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KumarSatyam24/AI-Based-Real-Time-Intruder-Detection-System-Using-Face-Recognition-and-Telegram-Alerts.git
    cd facial_detection_system
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *The YOLOv8 model (`yolov8n.pt`) will be automatically downloaded on the first run.*

## 🚀 Usage / Quick Start

### 1. Place Videos
Put your video files (`.mp4`, `.avi`, etc.) into the `input_videos/` directory.

### 2. (Optional) Setup Telegram Notifications
Get real-time alerts on your phone! See **[Telegram Setup Guide](TELEGRAM_README.md)** for 5-minute setup.

```bash
# Quick setup:
pip install python-telegram-bot pillow
# Configure in config.py, then test:
python test_telegram_connection.py
```

### 3. Run Detection
Execute the main script from the root directory.

```bash
# Process all videos in the input folder
python run_intrusion_detection.py --all

# Process one or more specific videos
python run_intrusion_detection.py --select intruder_1.mp4,intruder_2.mp4

# Run with an interactive file picker
python run_intrusion_detection.py --select
```

### 4. View Results
-   **Annotated Videos**: Find the processed videos with bounding boxes in the `output_videos/` folder.
-   **Incident Reports**: Detailed JSON reports are saved in `data/incidents/`.
-   **Telegram Alerts**: Receive instant mobile notifications (if enabled).

## 🔧 Configuration & Tools

### Define Regions of Interest (ROIs)
For targeted damage detection, you must define ROIs for doors and windows.

```bash
python select_rois.py input_videos/your_video.mp4
```
**ROI Selector Controls:**
-   **Click and Drag**: Draw a rectangle.
-   **`d` key**: Mark the last drawn rectangle as a "door".
-   **`w` key**: Mark it as a "window".
-   **`u` key**: Undo the last ROI.
-   **`s` key**: Save all ROIs and exit.
-   **`q` key**: Quit without saving.

### Real-time Damage Tuning
Fine-tune damage detection parameters with a live preview.

```bash
python debug_damage_detection.py input_videos/your_video.mp4
```
Use the trackbars to adjust sensitivity and find the optimal values for your environment.

### Main Configuration File
All major parameters can be adjusted in `intrusion_detection/config.py`.

## 📊 Incident Reports

### View Reports from the CLI
Use the `view_incidents.py` script to parse and display reports.

```bash
# View the most recent incident report
python view_incidents.py

# List all available reports
python view_incidents.py --list

# View a specific report by file path
python view_incidents.py --file data/incidents/intruder_4_incidents_20251015_012121.json
```

### Report Structure
The system generates a concise JSON summary for each video.

```json
{
  "video_source": "intruder_4.mp4",
  "processing_timestamp": "2025-10-15T01:21:21",
  "total_frames_processed": 1787,
  "motion_summary": {
    "total_intrusion_periods": 1,
    "intrusion_periods": [
      {
        "period_id": 1,
        "start_time_seconds": 21.39,
        "end_time_seconds": 24.94,
        "duration_seconds": 3.56
      }
    ]
  },
  "damage_incidents": [
    {
      "timestamp_seconds": 45.12,
      "roi_type": "window",
      "bounding_box": [100, 150, 50, 50]
    }
  ]
}
```

## 📁 Project Structure

```
facial_detection_system/
├── intrusion_detection/    # Core Python package for all logic
├── data/                   # For generated data like reports and ROIs
│   ├── incidents/
│   └── rois/
├── input_videos/           # Place your source videos here
├── output_videos/          # Annotated videos are saved here
│
├── run_intrusion_detection.py  # Main script to run the system
├── select_rois.py              # Interactive ROI selection tool
├── debug_damage_detection.py   # Real-time damage tuning tool
├── view_incidents.py           # CLI tool to view reports
│
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/KumarSatyam24/AI-Based-Real-Time-Intruder-Detection-System-Using-Face-Recognition-and-Telegram-Alerts/issues).

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
⭐ *If you find this project useful, please consider giving it a star!*
