"""
Unit Tests for Video Capture Module
Epic 8: Testing & Validation - Story Point 25
"""

import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_capture import VideoCapture, VideoDisplay, FPSCalculator


class TestVideoCapture:
    """Test cases for VideoCapture class."""
    
    @pytest.fixture
    def video_capture(self):
        """Create VideoCapture instance for testing."""
        return VideoCapture(camera_source=0, frame_width=640, frame_height=480, frame_skip_rate=2)
    
    def test_video_capture_initialization(self, video_capture):
        """Test VideoCapture initialization."""
        assert video_capture.camera_source == 0
        assert video_capture.frame_width == 640
        assert video_capture.frame_height == 480
        assert video_capture.frame_skip_rate == 2
        assert video_capture.frame_count == 0
        assert video_capture.cap is None
        assert video_capture.is_connected is False
    
    @patch('cv2.VideoCapture')
    def test_initialize_camera_success(self, mock_cv2_videocapture, video_capture):
        """Test successful camera initialization."""
        # Mock cv2.VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cv2_videocapture.return_value = mock_cap
        
        # Test initialization
        result = video_capture.initialize_camera()
        
        assert result is True
        assert video_capture.is_connected is True
        assert video_capture.cap is not None
        
        # Verify cv2.VideoCapture was called with correct parameters
        mock_cv2_videocapture.assert_called_once_with(0)
        mock_cap.set.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_initialize_camera_failure(self, mock_cv2_videocapture, video_capture):
        """Test camera initialization failure."""
        # Mock cv2.VideoCapture to return unopened capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_videocapture.return_value = mock_cap
        
        # Test initialization
        result = video_capture.initialize_camera()
        
        assert result is False
        assert video_capture.is_connected is False
    
    def test_read_frame_without_initialization(self, video_capture):
        """Test reading frame without camera initialization."""
        result, frame = video_capture.read_frame()
        
        assert result is False
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_read_frame_with_frame_skipping(self, mock_cv2_videocapture, video_capture):
        """Test frame reading with frame skipping."""
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_videocapture.return_value = mock_cap
        
        # Initialize camera
        video_capture.initialize_camera()
        
        # Test frame skipping (frame_skip_rate = 2)
        # First frame should be skipped
        result, frame = video_capture.read_frame()
        assert result is True
        assert frame is None  # Frame was skipped
        
        # Second frame should be returned
        result, frame = video_capture.read_frame()
        assert result is True
        assert frame is not None
    
    def test_camera_properties(self, video_capture):
        """Test getting camera properties without initialization."""
        properties = video_capture.get_camera_properties()
        assert properties == {}
    
    @patch('cv2.VideoCapture')
    def test_camera_properties_with_initialization(self, mock_cv2_videocapture, video_capture):
        """Test getting camera properties with initialization."""
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_BRIGHTNESS: 0.5,
            cv2.CAP_PROP_CONTRAST: 0.5,
            cv2.CAP_PROP_SATURATION: 0.5
        }.get(prop, 0)
        mock_cv2_videocapture.return_value = mock_cap
        
        # Initialize camera
        video_capture.initialize_camera()
        
        # Get properties
        properties = video_capture.get_camera_properties()
        
        assert properties['width'] == 640
        assert properties['height'] == 480
        assert properties['fps'] == 30.0
    
    def test_context_manager(self, video_capture):
        """Test VideoCapture as context manager."""
        with patch('cv2.VideoCapture') as mock_cv2_videocapture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cv2_videocapture.return_value = mock_cap
            
            with video_capture as vc:
                assert vc.is_connected is True
            
            # Verify release was called
            mock_cap.release.assert_called_once()


class TestVideoDisplay:
    """Test cases for VideoDisplay class."""
    
    @pytest.fixture
    def video_display(self):
        """Create VideoDisplay instance for testing."""
        return VideoDisplay(window_name="Test Window")
    
    def test_video_display_initialization(self, video_display):
        """Test VideoDisplay initialization."""
        assert video_display.window_name == "Test Window"
        assert video_display.window_created is False
    
    @patch('cv2.namedWindow')
    def test_create_window(self, mock_named_window, video_display):
        """Test window creation."""
        video_display.create_window()
        
        assert video_display.window_created is True
        mock_named_window.assert_called_once_with("Test Window", cv2.WINDOW_AUTOSIZE)
    
    @patch('cv2.imshow')
    @patch('cv2.putText')
    def test_show_frame(self, mock_put_text, mock_imshow, video_display):
        """Test showing frame."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch.object(video_display, 'create_window'):
            video_display.show_frame(test_frame, fps=30.0)
        
        # Verify imshow was called
        mock_imshow.assert_called_once()
        
        # Verify text was added (FPS and timestamp)
        assert mock_put_text.call_count >= 2
    
    def test_show_frame_with_none(self, video_display):
        """Test showing None frame."""
        with patch('cv2.imshow') as mock_imshow:
            video_display.show_frame(None)
            mock_imshow.assert_not_called()
    
    @patch('cv2.waitKey')
    def test_wait_key(self, mock_wait_key, video_display):
        """Test wait key functionality."""
        mock_wait_key.return_value = ord('q')
        
        result = video_display.wait_key(1)
        
        assert result == ord('q')
        mock_wait_key.assert_called_once_with(1)


class TestFPSCalculator:
    """Test cases for FPSCalculator class."""
    
    @pytest.fixture
    def fps_calculator(self):
        """Create FPSCalculator instance for testing."""
        return FPSCalculator(buffer_size=5)
    
    def test_fps_calculator_initialization(self, fps_calculator):
        """Test FPSCalculator initialization."""
        assert len(fps_calculator.frame_times) == 0
        assert fps_calculator.buffer_size == 5
    
    def test_fps_update(self, fps_calculator):
        """Test FPS calculation."""
        with patch('video_capture.time.time') as mock_time:
            # Mock time progression - need to account for initial time in __init__
            time_values = [0.0, 0.0, 0.033, 0.066, 0.099, 0.132]  # ~30 FPS
            mock_time.side_effect = time_values
            
            # Create new calculator after mocking time
            fps_calc = FPSCalculator(buffer_size=5)
            
            fps_values = []
            for i in range(4):  # 4 updates
                fps = fps_calc.update()
                fps_values.append(fps)
            
            # Check that FPS calculation works
            assert len(fps_values) > 0
            # After multiple updates, should have reasonable FPS
            final_fps = fps_values[-1]
            assert final_fps > 0, f"Final FPS should be positive, got {final_fps}"
            assert 20.0 <= final_fps <= 50.0, f"FPS should be around 30, got {final_fps}"
    
    def test_buffer_size_limit(self, fps_calculator):
        """Test that buffer size is maintained."""
        with patch('time.time') as mock_time:
            # Generate more frame times than buffer size
            time_values = [i * 0.033 for i in range(10)]
            mock_time.side_effect = time_values
            
            for _ in range(9):
                fps_calculator.update()
            
            # Buffer should not exceed buffer_size
            assert len(fps_calculator.frame_times) <= fps_calculator.buffer_size


# Integration tests
class TestVideoIntegration:
    """Integration tests for video components."""
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    def test_video_capture_display_integration(self, mock_imshow, mock_videocapture):
        """Test integration between VideoCapture and VideoDisplay."""
        # Setup mocks
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_videocapture.return_value = mock_cap
        
        # Create components
        video_capture = VideoCapture(frame_skip_rate=1)  # No frame skipping for test
        video_display = VideoDisplay()
        
        # Initialize and read frame
        video_capture.initialize_camera()
        ret, frame = video_capture.read_frame()
        
        assert ret is True
        assert frame is not None
        
        # Display frame
        with patch.object(video_display, 'create_window'):
            video_display.show_frame(frame)
        
        # Verify display was called
        mock_imshow.assert_called_once()


# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Suppress OpenCV window creation during tests
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    # Create test directories if needed
    test_dirs = ['test_data', 'test_logs']
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    import shutil
    for directory in test_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)


if __name__ == "__main__":
    pytest.main([__file__])
