# Technical Documentation: Advanced Intrusion Detection System

This document provides a detailed technical overview of the Advanced Intrusion Detection System, covering its architecture, core components, and operational logic.

## 1. System Architecture

The system is designed with a modular architecture, centered around a main processing pipeline that integrates several specialized detectors. The core components are:

- **Video Processing Pipeline (`processing.py`)**: The central engine that reads video frames, orchestrates the detection modules, and handles output generation.
- **Configuration (`config.py`)**: A centralized file holding all tunable parameters, from detection thresholds to file paths.
- **Detection Modules**:
    - **Motion Detector (`motion.py`)**: Identifies movement in restricted areas using background subtraction.
    - **Damage Detector (`damage.py`)**: Detects potential forced entry events like window shattering using multi-heuristic analysis.
    - **Person Detection**: Uses YOLOv8 model to verify that detected motion is caused by humans.
- **Incident Logging (`incidents.py`)**: Records all detected events into a structured JSON report.
- **Utility Scripts**: A set of command-line tools for running the system, selecting regions of interest, and viewing reports.

The typical workflow is as follows:
1. A video frame is captured from the source.
2. The frame is passed to the **Motion Detector**. If significant motion is detected, the system proceeds.
3. The frame is then analyzed by the **Damage Detector** for signs of forced entry.
4. The **YOLO model** detects persons in the frame to verify human intrusion.
5. All significant events (motion, damage, person detection) are passed to the **Incident Recorder**.
6. The frame is annotated with bounding boxes and status information.
7. At the end of the process, the **Incident Recorder** generates a concise JSON summary of all events.

## 2. Detection Modules

### 2.1. Motion Detection (`motion.py`)

The motion detector uses a background subtraction algorithm to identify movement within user-defined Regions of Interest (ROIs).

- **Algorithm**: `cv2.createBackgroundSubtractorKNN()` is used to maintain a running model of the static background.
- **Process**:
    1. The current frame is compared to the background model to generate a foreground mask.
    2. Morphological operations (`cv2.dilate`) are applied to the mask to close gaps and form coherent objects.
    3. Contours are found in the mask. Contours larger than a configured minimum area (`MOTION_MIN_CONTOUR_AREA`) are considered significant motion events.
- **Filtering**: To reduce false positives from non-human sources, motion is only considered an "intrusion" if the YOLOv8 model detects a person within the bounding box of the motion.
- **Person Detection**: The system uses YOLOv8 (You Only Look Once) object detection model to identify persons in the frame, ensuring that only human-caused motion triggers alerts.

### 2.2. Damage Detection (`damage.py`)

This module is designed to detect high-energy events indicative of forced entry, such as breaking glass or smashing a door. It uses a combination of three heuristics applied to a specific ROI (e.g., a window).

- **Heuristics**:
    1.  **Difference Threshold**: It calculates the absolute difference between the current frame and an exponentially smoothed moving average of previous frames. A high difference (`DAMAGE_DIFF_THRESHOLD`) suggests a sudden, drastic change.
    2.  **Edge Spike Ratio**: It runs a Canny edge detector on the frame difference. A sudden increase in the number of edges (`DAMAGE_EDGE_SPIKE_RATIO`) often correlates with shattering or splintering.
    3.  **Fragment Count**: It analyzes the number of small, independent contours ("fragments") within the damage ROI. A high count (`DAMAGE_FRAGMENT_COUNT_THRESHOLD`) can indicate debris from an impact.
- **Event Trigger**: A damage event is flagged only if **all three** heuristics surpass their configured thresholds simultaneously.
- **Persistence**: To ensure events are visible, detected damage bounding boxes are held on-screen for a configurable number of frames (`DAMAGE_EVENT_HOLD_FRAMES`).

## 3. Incident Logging (`incidents.py`)

The `IncidentRecorder` class provides a robust mechanism for logging all system events.

- **Purpose**: To create a single, comprehensive JSON file that summarizes all detected incidents from a video processing session.
- **Process**:
    1.  The recorder is initialized at the start of a run.
    2.  Throughout the processing loop, `log_motion()` and `log_damage()` methods are called whenever an event occurs.
    3.  Instead of logging every single frame of motion, the recorder groups continuous motion events into **"intrusion periods"**. A new period starts after a configurable cooldown period with no motion (`MOTION_COOLDOWN_PERIOD`).
- **Output (`incidents.json`)**:
    - `video_source`: The input video file.
    - `processing_timestamp`: When the report was generated.
    - `damage_incidents`: A list of all detected damage events, with timestamps and bounding box coordinates.
    - `motion_summary`: A concise list of intrusion periods, each with a start time, end time, and duration in seconds.

## 4. Configuration (`config.py`)

All system parameters are managed in `intrusion_detection/config.py`. This allows for easy tuning without modifying the core logic. Key parameters include:

- **File Paths**: Paths to the ROI file, YOLO model, output directories, etc.
- **Motion Detection**: `MOTION_MIN_CONTOUR_AREA`, `MOTION_COOLDOWN_PERIOD`.
- **Damage Detection**: `DAMAGE_DIFF_THRESHOLD`, `DAMAGE_EDGE_SPIKE_RATIO`, `DAMAGE_FRAGMENT_COUNT_THRESHOLD`, `DAMAGE_EVENT_HOLD_FRAMES`.
- **Person Detection**: `YOLO_CONFIDENCE_THRESHOLD`, `YOLO_MODEL_PATH`.
- **Logging**: `ENABLE_INCIDENT_LOGGING`, `INCIDENT_LOG_DIR`.

## 5. Scripts and Usage

The system includes several scripts for operation:

- **`run_intrusion_detection.py`**: The main script to run the full detection pipeline on a video file.
- **`select_roi.py`**: A utility to draw and save the motion and damage ROIs for a given video.
- **`debug_damage_detection.py`**: A debugging tool with interactive trackbars to fine-tune damage detection parameters in real-time.
- **`view_incidents.py`**: A tool to parse and display the contents of an `incidents.json` report in a human-readable format.

## 6. Technical Implementation Details

### 6.1. Background Subtraction

The system uses OpenCV's KNN (K-Nearest Neighbors) background subtractor, which models each pixel as a mixture of Gaussians. This approach is robust to gradual lighting changes and can handle minor background movements like swaying trees.

### 6.2. Multi-Heuristic Damage Detection

The damage detection algorithm combines three independent signals:
- **Intensity Change**: Measures raw pixel differences to detect sudden visual changes.
- **Edge Activity**: Analyzes the structure of changes to identify splintering or shattering patterns.
- **Fragmentation**: Counts discrete objects to detect debris or broken pieces.

This multi-signal approach significantly reduces false positives compared to single-metric systems.

### 6.3. Persistent Event Visualization

To improve operator awareness, detected damage events are "held" on screen for multiple frames. The system maintains a time-stamped queue of recent events and continues to render their bounding boxes until they expire.

### 6.4. Efficient Incident Summarization

Rather than logging every frame, the incident recorder groups consecutive motion frames into "intrusion periods." This reduces report size by ~90% while preserving all critical information about when intrusions occurred and their duration.
