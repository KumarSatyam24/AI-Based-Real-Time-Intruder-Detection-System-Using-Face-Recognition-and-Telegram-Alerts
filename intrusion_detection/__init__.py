"""
Intrusion Detection System package.

Modules:
- config: configuration, directories, logging, YOLO init
- video: VideoCapture wrapper
- background: background subtractors (KNN/MOG2)
- motion: MotionDetector with YOLO-based person filtering
- alerts: SimpleAlertSystem for logging/notifications with Telegram integration
- telegram_notifier: TelegramNotifier for real-time alert delivery
- system: SimpleMotionDetectionSystem orchestrating components
- processing: imageio-based enhanced video creation and batch processing
- interactive: helpers for interactive video selection in notebooks
"""

from .config import Config, setup_directories, setup_logging, init_yolo, get_logger
from .video import VideoCapture
from .background import BackgroundSubtractor
from .motion import MotionDetector
from .alerts import SimpleAlertSystem
from .processing import create_full_enhanced_video_with_imageio, process_all_input_videos
from .system import SimpleMotionDetectionSystem
from .interactive import list_input_videos, prompt_video_selection
from .damage import DamageDetector, DamageEvent
from .roi_selector import select_rois_for_video, load_rois_from_file, save_rois_to_file
from .incidents import IncidentRecorder, MotionIncident, DamageIncident, generate_summary_report

# Telegram notifier is optional
try:
    from .telegram_notifier import TelegramNotifier
    __all_telegram__ = ['TelegramNotifier']
except ImportError:
    __all_telegram__ = []

__all__ = [
    'Config', 'setup_directories', 'setup_logging', 'init_yolo', 'get_logger',
    'VideoCapture', 'BackgroundSubtractor', 'MotionDetector', 'SimpleAlertSystem',
    'create_full_enhanced_video_with_imageio', 'process_all_input_videos',
    'SimpleMotionDetectionSystem', 'list_input_videos', 'prompt_video_selection',
    'DamageDetector', 'DamageEvent', 'select_rois_for_video', 'load_rois_from_file', 'save_rois_to_file',
    'IncidentRecorder', 'MotionIncident', 'DamageIncident', 'generate_summary_report'
] + __all_telegram__