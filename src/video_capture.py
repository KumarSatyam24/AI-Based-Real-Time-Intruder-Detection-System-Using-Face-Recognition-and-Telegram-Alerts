"""
Video Capture Module
Epic 2: Video Capture Module - Story Points 4, 5

Handles video capture from webcams or CCTV cameras with optimization features.
"""

import cv2
import time
from typing import Optional, Tuple
from loguru import logger


class VideoCapture:
    """
    Video capture class for handling live video streams with optimization features.
    """
    
    def __init__(self, camera_source: int = 0, frame_width: int = 640, 
                 frame_height: int = 480, frame_skip_rate: int = 2):
        """
        Initialize video capture.
        
        Args:
            camera_source: Camera index (0 for default webcam) or IP camera URL
            frame_width: Width of captured frames
            frame_height: Height of captured frames
            frame_skip_rate: Process every nth frame (optimization)
        """
        self.camera_source = camera_source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_skip_rate = frame_skip_rate
        self.frame_count = 0
        self.cap = None
        self.is_connected = False
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera connection.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False
            
            # Set frame dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_connected = True
            logger.info(f"Camera initialized successfully: {self.camera_source}")
            logger.info(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the video source with frame skipping optimization.
        
        Returns:
            Tuple[bool, Optional[cv2.Mat]]: (success, frame) - success flag and frame data
        """
        if not self.is_connected or self.cap is None:
            logger.warning("Camera not initialized or disconnected")
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                self.is_connected = False
                return False, None
            
            self.frame_count += 1
            
            # Frame skipping for optimization
            if self.frame_count % self.frame_skip_rate != 0:
                return True, None  # Skip this frame
            
            # Downscale frame for better performance if needed
            if frame.shape[1] > 640:  # If width > 640
                scale_factor = 640 / frame.shape[1]
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return False, None
    
    def is_camera_connected(self) -> bool:
        """
        Check if camera is still connected.
        
        Returns:
            bool: True if camera is connected, False otherwise
        """
        return self.is_connected and self.cap is not None and self.cap.isOpened()
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect to camera...")
        self.release()
        time.sleep(2)  # Wait before reconnection attempt
        return self.initialize_camera()
    
    def get_camera_properties(self) -> dict:
        """
        Get current camera properties.
        
        Returns:
            dict: Dictionary containing camera properties
        """
        if not self.is_connected or self.cap is None:
            return {}
        
        properties = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
        
        return properties
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera resources released")
        self.is_connected = False
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class VideoDisplay:
    """
    Helper class for displaying video frames with annotations.
    """
    
    def __init__(self, window_name: str = "Facial Detection System"):
        """
        Initialize video display.
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.window_created = False
    
    def create_window(self):
        """Create display window."""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
    
    def show_frame(self, frame: cv2.Mat, fps: Optional[float] = None):
        """
        Display frame with optional FPS counter.
        
        Args:
            frame: Frame to display
            fps: Current FPS to display
        """
        if frame is None:
            return
        
        display_frame = frame.copy()
        
        # Add FPS counter if provided
        if fps is not None:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.create_window()
        cv2.imshow(self.window_name, display_frame)
    
    def close_window(self):
        """Close display window."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
    
    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            int: Key code pressed
        """
        return cv2.waitKey(delay) & 0xFF


# FPS Calculator utility
class FPSCalculator:
    """
    Utility class for calculating FPS.
    """
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize FPS calculator.
        
        Args:
            buffer_size: Number of frames to average over
        """
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS calculation.
        
        Returns:
            float: Current FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return 0
