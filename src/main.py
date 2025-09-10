"""
Main Facial Detection System Application
Integrates all modules to create the complete facial detection and alert system.
"""

import asyncio
import cv2
import yaml
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path

# Import our modules
from video_capture import VideoCapture, VideoDisplay, FPSCalculator
from face_detection import FaceDetector, FaceAnnotator
from face_recognition_module import FaceDatabase, FaceRecognizer
from telegram_alerts import TelegramAlertSystem, IntruderImageManager
from database_management import DatabaseManager, DatabaseCLI

# Add environment variable support
from dotenv import load_dotenv


class FacialDetectionSystem:
    """
    Main facial detection system class that orchestrates all components.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the facial detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        
        # Initialize components
        self.video_capture = None
        self.video_display = None
        self.face_detector = None
        self.face_database = None
        self.face_recognizer = None
        self.telegram_alerts = None
        self.intruder_manager = None
        self.fps_calculator = None
        self.annotator = None
        
        # Performance metrics
        self.frame_count = 0
        self.detection_count = 0
        self.recognition_count = 0
        self.alert_count = 0
        
        # Initialize system
        self._initialize_logging()
        self._initialize_components()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Load environment variables
            env_path = Path("config/.env")
            if env_path.exists():
                load_dotenv(env_path)
            
            # Load YAML config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Override with environment variables if available
            telegram_config = config.get('telegram', {})
            telegram_config['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', telegram_config.get('bot_token', ''))
            telegram_config['chat_id'] = os.getenv('TELEGRAM_CHAT_ID', telegram_config.get('chat_id', ''))
            
            # Override camera source if specified
            camera_config = config.get('camera', {})
            camera_source = os.getenv('CAMERA_SOURCE')
            if camera_source:
                try:
                    camera_config['source'] = int(camera_source)
                except ValueError:
                    camera_config['source'] = camera_source  # Might be an IP camera URL
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _initialize_logging(self):
        """Initialize logging system."""
        try:
            log_config = self.config.get('logging', {})
            
            # Remove default logger
            logger.remove()
            
            # Add console logger if enabled
            if log_config.get('console_output', True):
                logger.add(
                    sys.stdout,
                    level=log_config.get('level', 'INFO'),
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
                )
            
            # Add file logger
            log_file = log_config.get('log_file', 'logs/facial_detection.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            logger.add(
                log_file,
                level=log_config.get('level', 'INFO'),
                rotation=log_config.get('max_log_size', '10MB'),
                retention=log_config.get('backup_count', 5),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
            
            logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing facial detection system components...")
        
        try:
            # Initialize video capture
            camera_config = self.config.get('camera', {})
            self.video_capture = VideoCapture(
                camera_source=camera_config.get('source', 0),
                frame_width=camera_config.get('frame_width', 640),
                frame_height=camera_config.get('frame_height', 480),
                frame_skip_rate=camera_config.get('frame_skip_rate', 2)
            )
            
            # Initialize video display
            system_config = self.config.get('system', {})
            if system_config.get('show_video_feed', True):
                self.video_display = VideoDisplay(
                    window_name=system_config.get('window_name', 'Facial Detection System')
                )
            
            # Initialize face detector
            detection_config = self.config.get('face_detection', {})
            self.face_detector = FaceDetector(
                method="haar",  # Start with haar for better performance
                confidence_threshold=0.5
            )
            
            # Initialize face database
            db_config = self.config.get('database', {})
            self.face_database = FaceDatabase(
                authorized_faces_dir=db_config.get('authorized_faces_dir', 'data/authorized_faces'),
                encodings_file=db_config.get('encodings_file', 'data/face_encodings.pkl')
            )
            
            # Initialize face recognizer
            recognition_config = self.config.get('face_recognition', {})
            self.face_recognizer = FaceRecognizer(
                database=self.face_database,
                tolerance=recognition_config.get('tolerance', 0.6),
                model=recognition_config.get('model', 'hog')
            )
            
            # Initialize Telegram alerts
            telegram_config = self.config.get('telegram', {})
            self.telegram_alerts = TelegramAlertSystem(
                bot_token=telegram_config.get('bot_token', ''),
                chat_id=telegram_config.get('chat_id', ''),
                alert_cooldown=telegram_config.get('alert_cooldown', 30),
                enable_alerts=telegram_config.get('enable_alerts', True)
            )
            
            # Initialize intruder image manager
            self.intruder_manager = IntruderImageManager(
                captured_intruders_dir=db_config.get('captured_intruders_dir', 'data/captured_intruders')
            )
            
            # Initialize utilities
            self.fps_calculator = FPSCalculator()
            self.annotator = FaceAnnotator()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)
    
    async def start_system(self):
        """Start the facial detection system."""
        logger.info("Starting Facial Detection System...")
        
        try:
            # Initialize camera
            if not self.video_capture.initialize_camera():
                logger.error("Failed to initialize camera")
                return False
            
            # Send startup notification
            await self.telegram_alerts.send_system_status(
                "startup",
                {"camera_initialized": True, "database_loaded": len(self.face_database.list_all_people())}
            )
            
            self.running = True
            logger.info("Facial Detection System started successfully")
            
            # Start main detection loop
            await self._main_detection_loop()
            
        except KeyboardInterrupt:
            logger.info("System stopped by user")
        except Exception as e:
            logger.error(f"Error in main system: {e}")
            await self.telegram_alerts.send_system_status("error", {"error": str(e)})
        finally:
            await self.stop_system()
        
        return True
    
    async def _main_detection_loop(self):
        """Main detection and recognition loop."""
        logger.info("Starting main detection loop...")
        
        last_detection_time = time.time()
        detection_interval = self.config.get('performance', {}).get('detection_interval', 0.1)
        
        while self.running:
            try:
                # Read frame from camera
                ret, frame = self.video_capture.read_frame()
                
                if not ret:
                    if not self.video_capture.is_camera_connected():
                        logger.warning("Camera disconnected, attempting reconnection...")
                        if not self.video_capture.reconnect():
                            logger.error("Failed to reconnect camera")
                            break
                    continue
                
                if frame is None:
                    continue  # Frame was skipped
                
                self.frame_count += 1
                current_time = time.time()
                
                # Perform face detection at specified interval
                if current_time - last_detection_time >= detection_interval:
                    await self._process_frame(frame, current_time)
                    last_detection_time = current_time
                
                # Update FPS
                fps = self.fps_calculator.update()
                
                # Display frame if enabled
                if self.video_display:
                    display_frame = frame.copy()
                    
                    # Add system info to display
                    self._add_system_info_to_frame(display_frame, fps)
                    
                    self.video_display.show_frame(display_frame, fps)
                    
                    # Check for quit key
                    if self.video_display.wait_key(1) == ord('q'):
                        logger.info("Quit key pressed")
                        self.running = False
                        break
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _process_frame(self, frame, current_time: float):
        """
        Process frame for face detection and recognition.
        
        Args:
            frame: Input frame
            current_time: Current timestamp
        """
        try:
            # Detect faces
            detection_start = time.time()
            face_locations = self.face_detector.detect_faces(frame)
            detection_time = time.time() - detection_start
            
            if face_locations:
                self.detection_count += len(face_locations)
                
                # Perform face recognition
                recognition_start = time.time()
                recognition_results = self.face_recognizer.recognize_faces(frame, face_locations)
                recognition_time = time.time() - recognition_start
                
                self.recognition_count += len(recognition_results)
                
                # Process recognition results
                await self._handle_recognition_results(
                    frame, face_locations, recognition_results, 
                    detection_time, recognition_time
                )
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    async def _handle_recognition_results(self, frame, face_locations, recognition_results, 
                                        detection_time: float, recognition_time: float):
        """
        Handle face recognition results and trigger alerts if necessary.
        
        Args:
            frame: Input frame
            face_locations: Detected face locations
            recognition_results: Recognition results
            detection_time: Time taken for detection
            recognition_time: Time taken for recognition
        """
        unknown_faces = []
        known_faces = []
        
        for i, (location, result) in enumerate(zip(face_locations, recognition_results)):
            if result['is_known']:
                known_faces.append((location, result))
                logger.debug(f"Known person detected: {result['name']} (confidence: {result['confidence']:.2f})")
            else:
                unknown_faces.append((location, result))
                logger.warning(f"Unknown person detected (confidence: {result['confidence']:.2f})")
        
        # Handle unknown faces
        if unknown_faces:
            await self._handle_unknown_faces(frame, unknown_faces)
        
        # Update display with annotations if enabled
        if self.video_display and self.config.get('system', {}).get('show_bounding_boxes', True):
            self._annotate_frame_with_results(frame, face_locations, recognition_results)
    
    async def _handle_unknown_faces(self, frame, unknown_faces):
        """
        Handle detection of unknown faces.
        
        Args:
            frame: Input frame
            unknown_faces: List of unknown face detections
        """
        for location, result in unknown_faces:
            x, y, w, h = location
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Save intruder image if enabled
            if self.config.get('performance', {}).get('save_intruder_images', True):
                self.intruder_manager.save_intruder_image(
                    face_region, 
                    confidence=result['confidence']
                )
            
            # Send Telegram alert
            additional_info = {
                'detection_method': 'face_recognition',
                'face_location': f"x:{x}, y:{y}, w:{w}, h:{h}",
                'system_fps': f"{self.fps_calculator.update():.1f}"
            }
            
            await self.telegram_alerts.send_intruder_alert(
                face_region,
                confidence=result['confidence'],
                additional_info=additional_info
            )
            
            self.alert_count += 1
    
    def _annotate_frame_with_results(self, frame, face_locations, recognition_results):
        """
        Annotate frame with face detection and recognition results.
        
        Args:
            frame: Input frame to annotate
            face_locations: Detected face locations
            recognition_results: Recognition results
        """
        if not (self.config.get('system', {}).get('show_bounding_boxes', True)):
            return
        
        labels = []
        face_types = []
        
        for result in recognition_results:
            name = result['name']
            confidence = result['confidence']
            
            if result['is_known']:
                labels.append(f"{name} ({confidence:.2f})")
                face_types.append('known')
            else:
                labels.append(f"Unknown ({confidence:.2f})")
                face_types.append('unknown')
        
        # Draw annotations
        annotated_frame = self.annotator.draw_face_boxes(
            frame, face_locations, labels, face_types
        )
        
        # Copy annotations back to original frame
        frame[:] = annotated_frame[:]
    
    def _add_system_info_to_frame(self, frame, fps: float):
        """
        Add system information overlay to frame.
        
        Args:
            frame: Frame to annotate
            fps: Current FPS
        """
        # System status text
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Alerts: {self.alert_count}",
            f"People in DB: {len(self.face_database.list_all_people())}"
        ]
        
        # Draw background rectangle
        overlay_height = len(status_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (250, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, overlay_height), (255, 255, 255), 1)
        
        # Draw text
        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    async def stop_system(self):
        """Stop the facial detection system."""
        logger.info("Stopping Facial Detection System...")
        
        self.running = False
        
        # Send shutdown notification
        stats = self.get_system_statistics()
        await self.telegram_alerts.send_system_status("shutdown", stats)
        
        # Release resources
        if self.video_capture:
            self.video_capture.release()
        
        if self.video_display:
            self.video_display.close_window()
        
        cv2.destroyAllWindows()
        
        logger.info("Facial Detection System stopped")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        return {
            'frames_processed': self.frame_count,
            'faces_detected': self.detection_count,
            'recognitions_performed': self.recognition_count,
            'alerts_sent': self.alert_count,
            'people_in_database': len(self.face_database.list_all_people()),
            'current_fps': self.fps_calculator.update() if self.fps_calculator else 0,
            'telegram_alerts_enabled': self.telegram_alerts.enable_alerts if self.telegram_alerts else False,
            'camera_connected': self.video_capture.is_camera_connected() if self.video_capture else False
        }


async def main():
    """Main entry point for the facial detection system."""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Facial Detection System')
    parser.add_argument('--config', default='config/settings.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--test-telegram', action='store_true',
                       help='Send test Telegram message and exit')
    parser.add_argument('--database-manager', action='store_true',
                       help='Run database manager CLI')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = FacialDetectionSystem(config_path=args.config)
        
        if args.test_telegram:
            # Test Telegram connection
            success = await system.telegram_alerts.send_test_message()
            if success:
                print("✅ Telegram test message sent successfully!")
            else:
                print("❌ Failed to send Telegram test message.")
            return
        
        if args.database_manager:
            # Run database manager CLI
            db_manager = DatabaseManager(
                system.face_database,
                system.config.get('database', {}).get('authorized_faces_dir', 'data/authorized_faces')
            )
            cli = DatabaseCLI(db_manager)
            cli.run_interactive_mode()
            return
        
        # Start main system
        await system.start_system()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
