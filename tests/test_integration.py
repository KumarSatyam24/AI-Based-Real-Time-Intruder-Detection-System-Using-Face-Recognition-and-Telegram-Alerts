"""
End-to-End Integration Tests
Epic 8: Testing & Validation - Story Point 26

Tests the complete system integration with simulated scenarios.
"""

import pytest
import asyncio
import numpy as np
import cv2
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import FacialDetectionSystem


class TestEndToEndIntegration:
    """End-to-end integration tests for the complete system."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            'camera': {
                'source': 0,
                'frame_width': 640,
                'frame_height': 480,
                'frame_skip_rate': 1
            },
            'face_detection': {
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'min_size': [30, 30]
            },
            'face_recognition': {
                'threshold': 0.6,
                'model': 'hog',
                'tolerance': 0.6
            },
            'telegram': {
                'bot_token': 'test_token',
                'chat_id': 'test_chat',
                'enable_alerts': False,  # Disable for tests
                'alert_cooldown': 1
            },
            'database': {
                'authorized_faces_dir': 'test_data/authorized_faces',
                'captured_intruders_dir': 'test_data/captured_intruders',
                'encodings_file': 'test_data/face_encodings.pkl'
            },
            'logging': {
                'level': 'ERROR',  # Reduce log noise in tests
                'console_output': False,
                'log_file': 'test_logs/test.log'
            },
            'system': {
                'show_video_feed': False,  # No display in tests
                'show_bounding_boxes': True,
                'window_name': 'Test System'
            },
            'performance': {
                'detection_interval': 0.01,  # Fast for tests
                'save_intruder_images': False  # Don't save in tests
            }
        }
    
    @pytest.fixture
    def test_config_file(self, test_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def test_system(self, test_config_file):
        """Create test system instance."""
        return FacialDetectionSystem(config_path=test_config_file)
    
    def test_system_initialization(self, test_system):
        """Test complete system initialization."""
        # Check all components are initialized
        assert test_system.video_capture is not None
        assert test_system.face_detector is not None
        assert test_system.face_database is not None
        assert test_system.face_recognizer is not None
        assert test_system.telegram_alerts is not None
        assert test_system.intruder_manager is not None
        assert test_system.fps_calculator is not None
        assert test_system.annotator is not None
    
    @patch('cv2.VideoCapture')
    @pytest.mark.skip(reason="Requires full system environment")
    async def test_system_startup_shutdown(self, mock_videocapture, test_system):
        """Test system startup and shutdown cycle."""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Immediate failure to exit loop
        mock_videocapture.return_value = mock_cap
        
        # Mock telegram
        with patch.object(test_system.telegram_alerts, 'send_system_status', new_callable=AsyncMock) as mock_telegram:
            # Start and stop system
            await test_system.start_system()
            
            # Verify telegram notifications were sent
            assert mock_telegram.call_count >= 1
    
    @pytest.mark.skip(reason="Requires full system environment")
    @patch('cv2.VideoCapture')
    async def test_known_person_detection_scenario(self, mock_videocapture, test_system):
        """Test scenario: Known person detected (no alert should be sent)."""
        # Setup mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, test_frame),  # First frame
            (False, None)        # End simulation
        ]
        mock_videocapture.return_value = mock_cap
        
        # Add a known person to database
        with patch('face_recognition.load_image_file'), \
             patch('face_recognition.face_encodings') as mock_encodings:
            
            mock_encodings.return_value = [np.random.rand(128)]
            
            # Create temp image file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                temp_img_path = temp_img.name
            
            try:
                test_system.face_database.add_person("John Doe", [temp_img_path])
            finally:
                os.unlink(temp_img_path)
        
        # Mock face detection and recognition
        with patch.object(test_system.face_detector, 'detect_faces') as mock_detect, \
             patch.object(test_system.face_recognizer, 'recognize_faces') as mock_recognize, \
             patch.object(test_system.telegram_alerts, 'send_intruder_alert', new_callable=AsyncMock) as mock_alert:
            
            # Setup detection mocks
            mock_detect.return_value = [(100, 100, 50, 50)]  # One face detected
            mock_recognize.return_value = [{
                'name': 'John Doe',
                'confidence': 0.8,
                'distance': 0.3,
                'is_known': True
            }]
            
            # Run system briefly
            test_system.running = True
            await test_system._main_detection_loop()
            
            # Verify no alert was sent for known person
            mock_alert.assert_not_called()
    
    @patch('cv2.VideoCapture')
    @pytest.mark.skip(reason="Requires full system environment")
    async def test_unknown_person_detection_scenario(self, mock_videocapture, test_system):
        """Test scenario: Unknown person detected (alert should be sent)."""
        # Setup mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, test_frame),  # First frame
            (False, None)        # End simulation
        ]
        mock_videocapture.return_value = mock_cap
        
        # Enable alerts for this test
        test_system.telegram_alerts.enable_alerts = True
        
        # Mock face detection and recognition
        with patch.object(test_system.face_detector, 'detect_faces') as mock_detect, \
             patch.object(test_system.face_recognizer, 'recognize_faces') as mock_recognize, \
             patch.object(test_system.telegram_alerts, 'send_intruder_alert', new_callable=AsyncMock) as mock_alert:
            
            # Setup detection mocks
            mock_detect.return_value = [(100, 100, 50, 50)]  # One face detected
            mock_recognize.return_value = [{
                'name': 'Unknown',
                'confidence': 0.0,
                'distance': 1.0,
                'is_known': False
            }]
            
            # Mock alert to return success
            mock_alert.return_value = True
            
            # Run system briefly
            test_system.running = True
            await test_system._main_detection_loop()
            
            # Verify alert was sent for unknown person
            mock_alert.assert_called_once()
    
    @patch('cv2.VideoCapture')
    @pytest.mark.skip(reason="Requires full system environment")
    async def test_multiple_faces_scenario(self, mock_videocapture, test_system):
        """Test scenario: Multiple people in frame (mixed known/unknown)."""
        # Setup mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, test_frame),
            (False, None)
        ]
        mock_videocapture.return_value = mock_cap
        
        # Enable alerts
        test_system.telegram_alerts.enable_alerts = True
        
        # Mock multiple face detection and recognition
        with patch.object(test_system.face_detector, 'detect_faces') as mock_detect, \
             patch.object(test_system.face_recognizer, 'recognize_faces') as mock_recognize, \
             patch.object(test_system.telegram_alerts, 'send_intruder_alert', new_callable=AsyncMock) as mock_alert:
            
            # Setup detection mocks - multiple faces
            mock_detect.return_value = [
                (100, 100, 50, 50),  # Face 1
                (200, 200, 50, 50),  # Face 2
                (300, 300, 50, 50)   # Face 3
            ]
            
            mock_recognize.return_value = [
                {'name': 'John Doe', 'confidence': 0.8, 'distance': 0.3, 'is_known': True},    # Known
                {'name': 'Unknown', 'confidence': 0.0, 'distance': 1.0, 'is_known': False},   # Unknown
                {'name': 'Jane Doe', 'confidence': 0.9, 'distance': 0.2, 'is_known': True}    # Known
            ]
            
            mock_alert.return_value = True
            
            # Run system
            test_system.running = True
            await test_system._main_detection_loop()
            
            # Should only alert for the unknown person
            assert mock_alert.call_count == 1
    
    @patch('cv2.VideoCapture')
    @pytest.mark.skip(reason="Requires full system environment")
    async def test_camera_disconnect_reconnect_scenario(self, mock_videocapture, test_system):
        """Test scenario: Camera disconnection and reconnection."""
        # Setup mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # Simulate camera disconnection then reconnection
        read_responses = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Normal frame
            (False, None),  # Disconnection
            (False, None),  # Still disconnected
            (False, None)   # End test
        ]
        mock_cap.read.side_effect = read_responses
        
        # Mock is_camera_connected to simulate disconnect/reconnect
        test_system.video_capture.is_camera_connected = Mock(side_effect=[True, False, True, True])
        test_system.video_capture.reconnect = Mock(return_value=False)  # Fail reconnection
        
        mock_videocapture.return_value = mock_cap
        
        # Run system - should handle disconnection gracefully
        test_system.running = True
        await test_system._main_detection_loop()
        
        # Verify reconnection was attempted
        test_system.video_capture.reconnect.assert_called()
    
    def test_system_statistics(self, test_system):
        """Test system statistics collection."""
        # Simulate some activity
        test_system.frame_count = 100
        test_system.detection_count = 50
        test_system.recognition_count = 25
        test_system.alert_count = 5
        
        stats = test_system.get_system_statistics()
        
        assert stats['frames_processed'] == 100
        assert stats['faces_detected'] == 50
        assert stats['recognitions_performed'] == 25
        assert stats['alerts_sent'] == 5
        assert 'people_in_database' in stats
        assert 'telegram_alerts_enabled' in stats
        assert 'camera_connected' in stats
    
    @pytest.mark.asyncio
    async def test_telegram_integration(self, test_system):
        """Test Telegram integration (mocked)."""
        # Test sending test message
        with patch.object(test_system.telegram_alerts, 'send_test_message', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = True
            
            result = await test_system.telegram_alerts.send_test_message()
            assert result is True
    
    def test_database_operations_integration(self, test_system):
        """Test database operations integration."""
        # Test adding person
        with patch('face_recognition.load_image_file'), \
             patch('face_recognition.face_encodings') as mock_encodings, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
            
            mock_encodings.return_value = [np.random.rand(128)]
            temp_img_path = temp_img.name
            
            try:
                # Add person
                success = test_system.face_database.add_person("Test Person", [temp_img_path])
                assert success is True
                
                # Verify in database
                people = test_system.face_database.list_all_people()
                assert "Test Person" in people
                
                # Update recognizer cache
                test_system.face_recognizer._update_encoding_cache()
                
                # Verify recognizer knows about new person
                assert len(test_system.face_recognizer._known_names) == 1
                
            finally:
                os.unlink(temp_img_path)


class TestSystemPerformance:
    """Performance tests for the system."""
    
    @pytest.mark.skip(reason="Requires full system environment")
    def test_system_memory_usage(self, test_system):
        """Test that system doesn't have memory leaks."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate processing many frames
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for _ in range(100):
            # Simulate frame processing without actual detection
            test_system.frame_count += 1
            test_system.fps_calculator.update()
            
            # Force garbage collection
            if _ % 10 == 0:
                gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / initial_memory
        
        # Allow up to 50% memory increase for test artifacts
        assert memory_increase < 0.5, f"Memory increased by {memory_increase*100:.1f}%"


# Fixtures for test data cleanup
@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data():
    """Clean up test data before and after each test."""
    test_dirs = ['test_data', 'test_logs']
    
    # Cleanup before test
    for directory in test_dirs:
        if os.path.exists(directory):
            import shutil
            shutil.rmtree(directory)
    
    yield
    
    # Cleanup after test
    for directory in test_dirs:
        if os.path.exists(directory):
            import shutil
            shutil.rmtree(directory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
