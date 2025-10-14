import os
import datetime
import logging

try:
	from ultralytics import YOLO  # type: ignore
	YOLO_AVAILABLE = True
except Exception:
	YOLO_AVAILABLE = False
	YOLO = None  # type: ignore


class Config:
    # Folder Structure
    INPUT_VIDEO_DIR = './input_videos/'
    OUTPUT_VIDEO_DIR = './output_videos/'
    DATA_DIR = './data/'

    # Video Settings
    VIDEO_SOURCE = './input_videos/intruder_2.mp4'
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PROCESS_EVERY_N_FRAMES = 3

    # Motion Detection Settings
    MIN_CONTOUR_AREA_PERCENT = 0.015
    DETECTION_PERSISTENCE = 3

    # Person Detection Settings (ENHANCED)
    ENABLE_PERSON_DETECTION = True
    PERSON_ONLY_MODE = True  # If True, only track motion when person(s) detected
    PERSON_CONFIDENCE_THRESHOLD = 0.5
    MIN_PERSON_AREA = 1000
    MAX_PERSON_AREA = 200000
    PERSON_ASPECT_RATIO_MIN = 0.3
    PERSON_ASPECT_RATIO_MAX = 4.0
    MIN_OVERLAP_RATIO = 0.1

    # Grayscale Enhancement Settings (NEW)
    AUTO_CONVERT_GRAYSCALE = True
    ENHANCE_CONTRAST = True
    CLAHE_CLIP_LIMIT = 1.5
    CLAHE_TILE_SIZE = 8
    APPLY_GAMMA_CORRECTION = True
    GAMMA = 1.2  # >1.0 brightens output
    APPLY_GAMMA_TO_OUTPUT = True  # apply gamma on the final frame to reduce dark shading

    # Denoising/Pre-blur before background subtraction
    DENOISE_BEFORE_BG = True
    DENOISE_KERNEL_SIZE = 3  # must be odd
    DENOISE_SIGMA = 0

    # Damage / Structural Change Detection (heuristic)
    ENABLE_DAMAGE_DETECTION = True
    # Define door/window ROIs as list of (x, y, w, h) in resized frame coordinates
    DOOR_ROIS: list[tuple[int, int, int, int]] = []  # user can populate
    WINDOW_ROIS: list[tuple[int, int, int, int]] = []  # user can populate
    # --- Damage detection calibrated defaults (user-selected) ---
    DAMAGE_DIFF_THRESHOLD = 40.0  # mean abs diff threshold to flag large structural change (was 40.0)
    DAMAGE_EDGE_SPIKE_RATIO = 0.5  # lowered from 2.5 per user request (may increase sensitivity)
    DAMAGE_FRAGMENT_COUNT_THRESHOLD = 4  # lowered from 25 per user request
    DAMAGE_MIN_PERSISTENCE = 2  # frames to persist before confirming (requested 2)
    DAMAGE_EVENT_COOLDOWN = 300  # frames between repeated events per ROI
    DAMAGE_EVENT_HOLD_FRAMES = 60  # NEW: highlight bounding box this many frames after event for visibility

    # Background Subtraction Settings
    BACKGROUND_HISTORY = 500
    BACKGROUND_THRESHOLD = 400

    # Recording Settings
    SAVE_DETECTED_MOTION = True
    SAVE_MOTION_DIR = './data/detected_motion/'
    SAVE_ALERT_VIDEO = True
    VIDEO_CLIP_DURATION = 10

    # Alert Settings
    ALERT_COOLDOWN = 30

    # Logging Settings
    LOG_FILE = 'motion_detection_log.csv'
    LOG_LEVEL = logging.INFO

    # Incident Recording Settings
    ENABLE_INCIDENT_LOGGING = True
    INCIDENT_LOG_DIR = './data/incidents/'
    INCIDENT_LOG_FORMAT = 'json'  # 'json' or 'csv'
    LOG_MOTION_EVENTS = True  # Log intrusion/motion detection events
    LOG_DAMAGE_EVENTS = True  # Log damage detection events

    # Display Settings
    SHOW_PROCESSED_FRAMES = True
    DASHBOARD_UPDATE_INTERVAL = 5

    @staticmethod
    def get_output_video_path(input_video_name: str, suffix: str = "enhanced") -> str:
        base_name = os.path.splitext(os.path.basename(input_video_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_{suffix}_{timestamp}.mp4"
        return os.path.join(Config.OUTPUT_VIDEO_DIR, output_filename)


def setup_directories(cfg: Config) -> list[str]:
    directories = [
        cfg.INPUT_VIDEO_DIR,
        cfg.OUTPUT_VIDEO_DIR,
        cfg.DATA_DIR,
        cfg.SAVE_MOTION_DIR,
        cfg.INCIDENT_LOG_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories


def setup_logging(cfg: Config) -> logging.Logger:
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("motion_detection_system.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger('MotionDetectionSystem')
    return logger


def init_yolo(cfg: Config):
    """Initialize YOLO model if available and enabled. Returns the detector or None."""
    if not (YOLO_AVAILABLE and cfg.ENABLE_PERSON_DETECTION):
        return None
    try:
        return YOLO('yolov8n.pt')  # nano model for speed
    except Exception:
        return None


def get_logger() -> logging.Logger:
    return logging.getLogger('MotionDetectionSystem')
