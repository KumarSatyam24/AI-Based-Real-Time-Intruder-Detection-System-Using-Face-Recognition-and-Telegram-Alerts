"""
Telegram Alert System Module
Epic 6: Real-Time Alert System - Story Points 18, 19, 20, 21

Handles Telegram bot integration for real-time alerts when unknown persons are detected.
"""

import asyncio
import os
import cv2
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
from telegram import Bot
from telegram.error import TelegramError, NetworkError, TimedOut
import numpy as np


class TelegramAlertSystem:
    """
    Telegram bot integration for sending real-time security alerts.
    """
    
    def __init__(self, bot_token: str, chat_id: str, 
                 alert_cooldown: int = 30, enable_alerts: bool = True):
        """
        Initialize Telegram alert system.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID for alerts
            alert_cooldown: Minimum seconds between alerts for same person
            enable_alerts: Whether alerts are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.alert_cooldown = alert_cooldown
        self.enable_alerts = enable_alerts
        
        # Bot instance
        self.bot = None
        
        # Alert tracking
        self.last_alert_times = {}  # Track last alert time for each unknown person
        self.alert_history = []     # Store alert history
        
        # Initialize bot
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not provided. Alerts disabled.")
            self.enable_alerts = False
            return
        
        try:
            self.bot = Bot(token=self.bot_token)
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enable_alerts = False
    
    async def send_test_message(self) -> bool:
        """
        Send a test message to verify bot connection.
        
        Returns:
            bool: True if test message sent successfully
        """
        if not self.enable_alerts or not self.bot:
            logger.warning("Telegram alerts not enabled")
            return False
        
        try:
            test_message = (
                "🤖 *Facial Detection System Test Alert*\n\n"
                f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "✅ System is operational and connected to Telegram.\n\n"
                "This is a test message to verify the alert system is working properly."
            )
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=test_message,
                parse_mode='Markdown'
            )
            
            logger.info("Test message sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send test message: {e}")
            return False
    
    async def send_intruder_alert(self, image: np.ndarray, 
                                 detection_time: Optional[datetime] = None,
                                 confidence: float = 0.0,
                                 additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send intruder detection alert with image.
        
        Args:
            image: Detected intruder image (BGR format)
            detection_time: Time of detection
            confidence: Detection confidence
            additional_info: Additional information to include
            
        Returns:
            bool: True if alert sent successfully
        """
        if not self.enable_alerts or not self.bot:
            logger.debug("Telegram alerts not enabled, skipping alert")
            return False
        
        # Check cooldown
        current_time = time.time()
        if self._is_in_cooldown():
            logger.debug("Alert in cooldown period, skipping")
            return False
        
        try:
            # Update last alert time
            self.last_alert_times['unknown'] = current_time
            
            # Prepare alert message
            detection_time = detection_time or datetime.now()
            
            alert_message = (
                "🚨 *SECURITY ALERT - Unknown Person Detected* 🚨\n\n"
                f"🕒 Time: {detection_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📍 Location: Facial Detection System\n"
                f"🎯 Confidence: {confidence*100:.1f}%\n\n"
                "⚠️ An unrecognized person has been detected by the security system.\n"
                "Please review the attached image and take appropriate action if necessary."
            )
            
            # Add additional info if provided
            if additional_info:
                alert_message += "\n\n📋 Additional Information:\n"
                for key, value in additional_info.items():
                    alert_message += f"• {key}: {value}\n"
            
            # Convert image to bytes for sending
            image_bytes = self._image_to_bytes(image)
            
            if image_bytes:
                # Send photo with caption
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=image_bytes,
                    caption=alert_message,
                    parse_mode='Markdown'
                )
            else:
                # Send text message only if image conversion fails
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=alert_message + "\n\n⚠️ Image could not be attached.",
                    parse_mode='Markdown'
                )
            
            # Record alert in history
            self._record_alert(detection_time, confidence, additional_info)
            
            logger.info("Intruder alert sent successfully")
            return True
            
        except NetworkError as e:
            logger.error(f"Network error sending alert: {e}")
            return await self._retry_send_alert(image, detection_time, confidence, additional_info)
            
        except TimedOut as e:
            logger.error(f"Timeout sending alert: {e}")
            return await self._retry_send_alert(image, detection_time, confidence, additional_info)
            
        except TelegramError as e:
            logger.error(f"Telegram error sending alert: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending alert: {e}")
            return False
    
    async def send_system_status(self, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send system status message.
        
        Args:
            status: Status message ("startup", "shutdown", "error", "info")
            details: Additional details to include
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.enable_alerts or not self.bot:
            return False
        
        try:
            status_icons = {
                'startup': '🟢',
                'shutdown': '🔴',
                'error': '⚠️',
                'info': 'ℹ️'
            }
            
            icon = status_icons.get(status, 'ℹ️')
            
            message = (
                f"{icon} *Facial Detection System Status*\n\n"
                f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📊 Status: {status.upper()}\n"
            )
            
            if details:
                message += "\n📋 Details:\n"
                for key, value in details.items():
                    message += f"• {key}: {value}\n"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"System status message sent: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send status message: {e}")
            return False
    
    def _is_in_cooldown(self) -> bool:
        """
        Check if alert is in cooldown period.
        
        Returns:
            bool: True if in cooldown period
        """
        current_time = time.time()
        last_alert_time = self.last_alert_times.get('unknown', 0)
        
        return (current_time - last_alert_time) < self.alert_cooldown
    
    def _image_to_bytes(self, image: np.ndarray, quality: int = 85) -> Optional[bytes]:
        """
        Convert OpenCV image to bytes for Telegram.
        
        Args:
            image: OpenCV image (BGR format)
            quality: JPEG quality (1-100)
            
        Returns:
            Image bytes or None if conversion fails
        """
        try:
            # Encode image as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, buffer = cv2.imencode('.jpg', image, encode_param)
            
            if success:
                return buffer.tobytes()
            else:
                logger.error("Failed to encode image")
                return None
                
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return None
    
    async def _retry_send_alert(self, image: np.ndarray, detection_time: Optional[datetime],
                               confidence: float, additional_info: Optional[Dict[str, Any]],
                               max_retries: int = 3) -> bool:
        """
        Retry sending alert with exponential backoff.
        
        Args:
            image: Image to send
            detection_time: Detection time
            confidence: Detection confidence
            additional_info: Additional information
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if eventually successful
        """
        for attempt in range(max_retries):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            try:
                # Simplified retry - just send text message
                detection_time = detection_time or datetime.now()
                
                retry_message = (
                    "🚨 *SECURITY ALERT - Unknown Person Detected* 🚨\n\n"
                    f"🕒 Time: {detection_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"🎯 Confidence: {confidence*100:.1f}%\n\n"
                    "⚠️ An unrecognized person has been detected.\n"
                    f"📶 Alert sent after {attempt + 1} retry attempts."
                )
                
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=retry_message,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Alert sent successfully after {attempt + 1} retries")
                return True
                
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
        
        logger.error("All retry attempts failed")
        return False
    
    def _record_alert(self, detection_time: datetime, confidence: float,
                     additional_info: Optional[Dict[str, Any]]):
        """
        Record alert in history.
        
        Args:
            detection_time: Time of detection
            confidence: Detection confidence
            additional_info: Additional information
        """
        alert_record = {
            'timestamp': detection_time.isoformat(),
            'confidence': confidence,
            'additional_info': additional_info or {},
            'alert_sent_time': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert_record)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert system statistics.
        
        Returns:
            Dictionary containing alert statistics
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count alerts today
        alerts_today = sum(
            1 for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= today_start
        )
        
        # Count alerts this week
        week_start = today_start - timedelta(days=now.weekday())
        alerts_this_week = sum(
            1 for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= week_start
        )
        
        return {
            'enabled': self.enable_alerts,
            'bot_configured': self.bot is not None,
            'total_alerts': len(self.alert_history),
            'alerts_today': alerts_today,
            'alerts_this_week': alerts_this_week,
            'cooldown_seconds': self.alert_cooldown,
            'last_alert': self.alert_history[-1]['timestamp'] if self.alert_history else None
        }
    
    def set_alert_cooldown(self, cooldown_seconds: int):
        """
        Set alert cooldown period.
        
        Args:
            cooldown_seconds: Cooldown period in seconds
        """
        self.alert_cooldown = max(0, cooldown_seconds)
        logger.info(f"Alert cooldown set to {self.alert_cooldown} seconds")
    
    def enable_disable_alerts(self, enable: bool):
        """
        Enable or disable alerts.
        
        Args:
            enable: Whether to enable alerts
        """
        self.enable_alerts = enable and self.bot is not None
        status = "enabled" if self.enable_alerts else "disabled"
        logger.info(f"Telegram alerts {status}")


class IntruderImageManager:
    """
    Manager for saving and organizing intruder images.
    """
    
    def __init__(self, captured_intruders_dir: str):
        """
        Initialize intruder image manager.
        
        Args:
            captured_intruders_dir: Directory to save intruder images
        """
        self.captured_intruders_dir = captured_intruders_dir
        os.makedirs(captured_intruders_dir, exist_ok=True)
    
    def save_intruder_image(self, image: np.ndarray, 
                           detection_time: Optional[datetime] = None,
                           confidence: float = 0.0) -> Optional[str]:
        """
        Save intruder image to disk.
        
        Args:
            image: Image to save (BGR format)
            detection_time: Detection timestamp
            confidence: Detection confidence
            
        Returns:
            Path to saved image or None if save failed
        """
        try:
            detection_time = detection_time or datetime.now()
            
            # Generate filename with timestamp
            timestamp_str = detection_time.strftime("%Y%m%d_%H%M%S")
            confidence_str = f"{confidence*100:.0f}" if confidence > 0 else "00"
            filename = f"intruder_{timestamp_str}_conf{confidence_str}.jpg"
            
            filepath = os.path.join(self.captured_intruders_dir, filename)
            
            # Save image with high quality
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Intruder image saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save intruder image: {e}")
            return None
    
    def cleanup_old_images(self, days_to_keep: int = 30):
        """
        Clean up old intruder images.
        
        Args:
            days_to_keep: Number of days to keep images
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            for filename in os.listdir(self.captured_intruders_dir):
                filepath = os.path.join(self.captured_intruders_dir, filename)
                
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.debug(f"Removed old intruder image: {filename}")
            
            logger.info(f"Cleaned up intruder images older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old images: {e}")
    
    def get_recent_images(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get list of recent intruder images.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of image information dictionaries
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_images = []
            
            for filename in os.listdir(self.captured_intruders_dir):
                filepath = os.path.join(self.captured_intruders_dir, filename)
                
                if os.path.isfile(filepath) and filename.lower().endswith('.jpg'):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time >= cutoff_time:
                        recent_images.append({
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': file_time.isoformat(),
                            'size_bytes': os.path.getsize(filepath)
                        })
            
            # Sort by timestamp (newest first)
            recent_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return recent_images
            
        except Exception as e:
            logger.error(f"Error getting recent images: {e}")
            return []
