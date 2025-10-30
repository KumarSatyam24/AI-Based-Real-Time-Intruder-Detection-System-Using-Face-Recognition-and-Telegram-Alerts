from __future__ import annotations

import time
from typing import Optional, List
import numpy as np
from .config import get_logger, Config

try:
    from .telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramNotifier = None


class SimpleAlertSystem:
    """Enhanced alert system with Telegram integration."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger()
        self.last_alert_time = 0.0
        
        # Initialize Telegram notifier if enabled
        self.telegram_notifier: Optional[TelegramNotifier] = None
        if TELEGRAM_AVAILABLE and cfg.TELEGRAM_ENABLED:
            try:
                self.telegram_notifier = TelegramNotifier(cfg)
                if self.telegram_notifier.enabled:
                    self.logger.info("Telegram notifications enabled")
                else:
                    self.logger.warning("Telegram notifier initialization failed")
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram notifier: {e}")

    def notify(self, message: str):
        """Send a basic text notification (legacy method)."""
        now = time.time()
        if now - self.last_alert_time >= self.cfg.ALERT_COOLDOWN:
            self.logger.info("ALERT: %s", message)
            self.last_alert_time = now
    
    def notify_motion_detected(
        self,
        frame_number: int,
        person_count: int = 0,
        bounding_boxes: List[List[int]] = None,
        frame_image: Optional[np.ndarray] = None
    ):
        """
        Send motion detection alert with Telegram support.
        
        Args:
            frame_number: Frame number where motion was detected
            person_count: Number of persons detected
            bounding_boxes: List of bounding boxes [x, y, w, h]
            frame_image: Optional frame image to attach
        """
        # Log to console
        event_type = "Person detected" if person_count > 0 else "Motion detected"
        self.notify(f"{event_type} at frame {frame_number}")
        
        # Send Telegram notification if enabled
        if self.telegram_notifier and self.telegram_notifier.enabled:
            self.telegram_notifier.notify_motion_detected(
                frame_number=frame_number,
                person_count=person_count,
                bounding_boxes=bounding_boxes,
                frame_image=frame_image
            )
    
    def notify_damage_detected(
        self,
        frame_number: int,
        roi_type: str,
        roi_index: int,
        event_type: str,
        score: float,
        frame_image: Optional[np.ndarray] = None
    ):
        """
        Send damage detection alert with Telegram support.
        
        Args:
            frame_number: Frame number where damage was detected
            roi_type: Type of ROI ('door' or 'window')
            roi_index: Index of the ROI
            event_type: Type of event ('forced_entry' or 'shatter')
            score: Detection confidence score
            frame_image: Optional frame image to attach
        """
        # Log to console
        self.notify(f"Damage detected: {roi_type} {event_type} at frame {frame_number}")
        
        # Send Telegram notification if enabled
        if self.telegram_notifier and self.telegram_notifier.enabled:
            self.telegram_notifier.notify_damage_detected(
                frame_number=frame_number,
                roi_type=roi_type,
                roi_index=roi_index,
                event_type=event_type,
                score=score,
                frame_image=frame_image
            )
    
    def notify_session_summary(
        self,
        video_file: str,
        total_frames: int,
        motion_incidents: int,
        damage_incidents: int,
        processing_duration: float
    ):
        """
        Send processing session summary.
        
        Args:
            video_file: Name of the processed video file
            total_frames: Total number of frames processed
            motion_incidents: Number of motion incidents
            damage_incidents: Number of damage incidents
            processing_duration: Processing time in seconds
        """
        summary = (
            f"Processing complete: {video_file} - "
            f"{total_frames} frames, {motion_incidents} motion events, "
            f"{damage_incidents} damage events in {processing_duration:.1f}s"
        )
        self.logger.info(summary)
        
        # Send Telegram notification if enabled
        if self.telegram_notifier and self.telegram_notifier.enabled:
            self.telegram_notifier.notify_session_summary(
                video_file=video_file,
                total_frames=total_frames,
                motion_incidents=motion_incidents,
                damage_incidents=damage_incidents,
                processing_duration=processing_duration
            )
    
    def notify_error(self, error_message: str):
        """Send error notification."""
        self.logger.error(f"ALERT ERROR: {error_message}")
        
        if self.telegram_notifier and self.telegram_notifier.enabled:
            self.telegram_notifier.notify_error(error_message)
    
    def shutdown(self):
        """Clean shutdown of alert system."""
        if self.telegram_notifier and self.telegram_notifier.enabled:
            self.telegram_notifier.notify_system_shutdown()
