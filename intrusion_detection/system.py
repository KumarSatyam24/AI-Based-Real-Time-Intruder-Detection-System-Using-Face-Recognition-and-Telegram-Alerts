from __future__ import annotations

from .config import Config, init_yolo
from .alerts import SimpleAlertSystem
from .processing import process_all_input_videos


class SimpleMotionDetectionSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.alerts = SimpleAlertSystem(cfg)
        self.person_detector = init_yolo(cfg)

    def run_batch(self, selection=None, max_frames_per_video=None):
        return process_all_input_videos(
            self.cfg, 
            person_detector=self.person_detector, 
            selection=selection, 
            max_frames_per_video=max_frames_per_video,
            alert_system=self.alerts
        )
