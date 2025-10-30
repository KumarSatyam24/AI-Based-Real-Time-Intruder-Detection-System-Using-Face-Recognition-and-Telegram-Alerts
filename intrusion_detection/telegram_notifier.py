"""Telegram notification system for intrusion detection alerts."""
from __future__ import annotations

import asyncio
import io
import time
from typing import Optional, List
from datetime import datetime

try:
    from telegram import Bot
    from telegram.error import TelegramError
    import numpy as np
    from PIL import Image
    import cv2
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    TelegramError = None

from .config import Config, get_logger


class TelegramNotifier:
    """
    Telegram notification system for sending real-time alerts.
    Supports text messages, images, and videos.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger()
        self.enabled = cfg.TELEGRAM_ENABLED and TELEGRAM_AVAILABLE
        self.bot: Optional[Bot] = None
        self.last_notification_time = 0.0
        self.notification_count = 0
        
        if not TELEGRAM_AVAILABLE:
            self.logger.warning(
                "Telegram library not available. Install with: pip install python-telegram-bot pillow"
            )
            self.enabled = False
            return
            
        if self.enabled:
            self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize the Telegram bot."""
        if not self.cfg.TELEGRAM_BOT_TOKEN:
            self.logger.error("Telegram bot token not configured. Telegram notifications disabled.")
            self.enabled = False
            return
            
        if not self.cfg.TELEGRAM_CHAT_ID:
            self.logger.error("Telegram chat ID not configured. Telegram notifications disabled.")
            self.enabled = False
            return
        
        try:
            self.bot = Bot(token=self.cfg.TELEGRAM_BOT_TOKEN)
            self.logger.info("Telegram bot initialized successfully")
            
            # Send startup notification
            if self.cfg.TELEGRAM_SEND_STARTUP_MESSAGE:
                asyncio.run(self._send_startup_notification())
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    async def _send_startup_notification(self):
        """Send a notification when the system starts."""
        try:
            message = (
                "🟢 *Intrusion Detection System Started*\n\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🎥 Video Source: {self.cfg.VIDEO_SOURCE}\n"
                f"👤 Person Detection: {'Enabled' if self.cfg.ENABLE_PERSON_DETECTION else 'Disabled'}\n"
                f"🚪 Damage Detection: {'Enabled' if self.cfg.ENABLE_DAMAGE_DETECTION else 'Disabled'}\n"
                f"🔔 Notifications: Active"
            )
            await self.bot.send_message(
                chat_id=self.cfg.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            self.logger.error(f"Failed to send startup notification: {e}")
    
    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since the last notification."""
        now = time.time()
        if now - self.last_notification_time < self.cfg.TELEGRAM_NOTIFICATION_COOLDOWN:
            return False
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if we've exceeded the rate limit."""
        if self.cfg.TELEGRAM_MAX_NOTIFICATIONS_PER_HOUR > 0:
            # Simple rate limiting (resets don't matter for basic implementation)
            if self.notification_count >= self.cfg.TELEGRAM_MAX_NOTIFICATIONS_PER_HOUR:
                return False
        return True
    
    def notify_motion_detected(
        self,
        frame_number: int,
        person_count: int = 0,
        bounding_boxes: List[List[int]] = None,
        frame_image: Optional[np.ndarray] = None
    ):
        """
        Send notification for motion/intrusion detection.
        
        Args:
            frame_number: Frame number where motion was detected
            person_count: Number of persons detected
            bounding_boxes: List of bounding boxes [x, y, w, h]
            frame_image: Optional frame image to attach
        """
        if not self.enabled or not self._check_cooldown():
            return
            
        if not self._check_rate_limit():
            self.logger.warning("Telegram notification rate limit reached")
            return
        
        try:
            # Build message
            event_type = "👤 *Person Detected*" if person_count > 0 else "🚨 *Motion Detected*"
            message = (
                f"{event_type}\n\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🎬 Frame: {frame_number}\n"
            )
            
            if person_count > 0:
                message += f"👥 Persons: {person_count}\n"
            
            if bounding_boxes:
                message += f"📍 Detections: {len(bounding_boxes)}\n"
            
            # Send message with optional image
            if self.cfg.TELEGRAM_SEND_IMAGES and frame_image is not None:
                asyncio.run(self._send_photo_message(message, frame_image))
            else:
                asyncio.run(self._send_text_message(message))
            
            self.last_notification_time = time.time()
            self.notification_count += 1
            self.logger.info(f"Telegram notification sent: Motion detected at frame {frame_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to send motion notification: {e}")
    
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
        Send notification for damage detection.
        
        Args:
            frame_number: Frame number where damage was detected
            roi_type: Type of ROI ('door' or 'window')
            roi_index: Index of the ROI
            event_type: Type of event ('forced_entry' or 'shatter')
            score: Detection confidence score
            frame_image: Optional frame image to attach
        """
        if not self.enabled or not self._check_cooldown():
            return
            
        if not self._check_rate_limit():
            self.logger.warning("Telegram notification rate limit reached")
            return
        
        try:
            # Build message with appropriate emoji
            emoji_map = {
                'forced_entry': '🚪💥',
                'shatter': '🪟💔',
            }
            emoji = emoji_map.get(event_type, '⚠️')
            
            message = (
                f"{emoji} *DAMAGE DETECTED*\n\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🎬 Frame: {frame_number}\n"
                f"📍 Location: {roi_type.capitalize()} #{roi_index + 1}\n"
                f"⚠️ Event: {event_type.replace('_', ' ').title()}\n"
                f"📊 Score: {score:.1f}\n"
            )
            
            # Send message with optional image
            if self.cfg.TELEGRAM_SEND_IMAGES and frame_image is not None:
                asyncio.run(self._send_photo_message(message, frame_image))
            else:
                asyncio.run(self._send_text_message(message))
            
            self.last_notification_time = time.time()
            self.notification_count += 1
            self.logger.info(f"Telegram notification sent: Damage detected at frame {frame_number}")
            
        except Exception as e:
            self.logger.error(f"Failed to send damage notification: {e}")
    
    def notify_session_summary(
        self,
        video_file: str,
        total_frames: int,
        motion_incidents: int,
        damage_incidents: int,
        processing_duration: float
    ):
        """
        Send a summary notification at the end of video processing.
        
        Args:
            video_file: Name of the processed video file
            total_frames: Total number of frames processed
            motion_incidents: Number of motion incidents detected
            damage_incidents: Number of damage incidents detected
            processing_duration: Processing time in seconds
        """
        if not self.enabled:
            return
        
        try:
            message = (
                "📊 *Processing Complete*\n\n"
                f"🎥 Video: {video_file}\n"
                f"🎬 Frames: {total_frames}\n"
                f"🚨 Motion Events: {motion_incidents}\n"
                f"⚠️ Damage Events: {damage_incidents}\n"
                f"⏱️ Duration: {processing_duration:.1f}s\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            asyncio.run(self._send_text_message(message))
            self.logger.info("Telegram session summary sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send session summary: {e}")
    
    async def _send_text_message(self, message: str):
        """Send a text message to the configured chat."""
        try:
            await self.bot.send_message(
                chat_id=self.cfg.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
    
    async def _send_photo_message(self, caption: str, image: np.ndarray):
        """
        Send a photo message with caption.
        
        Args:
            caption: Message caption
            image: OpenCV image (BGR format)
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Resize if too large
            max_size = (1280, 720)
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            bio = io.BytesIO()
            pil_image.save(bio, format='JPEG', quality=85)
            bio.seek(0)
            
            # Send photo
            await self.bot.send_photo(
                chat_id=self.cfg.TELEGRAM_CHAT_ID,
                photo=bio,
                caption=caption,
                parse_mode='Markdown'
            )
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to send photo: {e}")
    
    def notify_error(self, error_message: str):
        """
        Send an error notification.
        
        Args:
            error_message: Error description
        """
        if not self.enabled:
            return
        
        try:
            message = (
                "❌ *System Error*\n\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⚠️ {error_message}"
            )
            
            asyncio.run(self._send_text_message(message))
            self.logger.info("Telegram error notification sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {e}")
    
    def notify_system_shutdown(self):
        """Send a notification when the system is shutting down."""
        if not self.enabled or not self.cfg.TELEGRAM_SEND_SHUTDOWN_MESSAGE:
            return
        
        try:
            message = (
                "🔴 *Intrusion Detection System Stopped*\n\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📊 Total Notifications Sent: {self.notification_count}"
            )
            
            asyncio.run(self._send_text_message(message))
            self.logger.info("Telegram shutdown notification sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send shutdown notification: {e}")
