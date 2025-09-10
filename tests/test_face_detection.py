"""
Unit Tests for Face Detection Module
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

from face_detection import FaceDetector, FaceAnnotator


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    @pytest.fixture
    def face_detector(self):
        """Create FaceDetector instance for testing."""
        return FaceDetector(method="haar", confidence_threshold=0.5)
    
    def test_face_detector_initialization(self, face_detector):
        """Test FaceDetector initialization."""
        assert face_detector.method == "haar"
        assert face_detector.confidence_threshold == 0.5
    
    @patch('cv2.CascadeClassifier')
    @patch('os.path.exists')
    def test_haar_cascade_initialization(self, mock_exists, mock_cascade, face_detector):
        """Test Haar cascade initialization."""
        # Mock file existence and cascade classifier
        mock_exists.return_value = True
        mock_cascade_instance = Mock()
        mock_cascade_instance.empty.return_value = False
        mock_cascade.return_value = mock_cascade_instance
        
        # Initialize detector (this happens in __init__)
        face_detector._initialize_haar_cascade()
        
        assert face_detector.haar_cascade is not None
    
    @patch('cv2.CascadeClassifier')
    @patch('os.path.exists')
    def test_haar_cascade_initialization_failure(self, mock_exists, mock_cascade, face_detector):
        """Test Haar cascade initialization failure."""
        # Mock file not existing
        mock_exists.return_value = False
        
        # Initialize to None first
        face_detector.haar_cascade = None
        face_detector._initialize_haar_cascade()
        
        # Should remain None when file doesn't exist
        assert face_detector.haar_cascade is None
    
    def test_detect_faces_without_detector(self, face_detector):
        """Test face detection without initialized detector."""
        # Ensure detector is not initialized
        face_detector.haar_cascade = None
        face_detector.dnn_net = None
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = face_detector.detect_faces(test_frame)
        
        assert faces == []
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_haar(self, mock_cascade, face_detector):
        """Test Haar cascade face detection."""
        # Setup mock cascade
        mock_cascade_instance = Mock()
        mock_cascade_instance.empty.return_value = False
        mock_cascade_instance.detectMultiScale.return_value = np.array([[100, 100, 50, 50], [200, 200, 60, 60]])
        mock_cascade.return_value = mock_cascade_instance
        
        face_detector.haar_cascade = mock_cascade_instance
        
        # Test detection
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = face_detector.detect_faces_haar(test_frame)
        
        assert len(faces) == 2
        assert faces[0] == (100, 100, 50, 50)
        assert faces[1] == (200, 200, 60, 60)
    
    def test_detect_faces_with_none_frame(self, face_detector):
        """Test face detection with None frame."""
        faces = face_detector.detect_faces(None)
        assert faces == []
    
    def test_detect_faces_with_empty_frame(self, face_detector):
        """Test face detection with empty frame."""
        empty_frame = np.array([])
        faces = face_detector.detect_faces(empty_frame)
        assert faces == []
    
    def test_remove_duplicate_faces(self, face_detector):
        """Test duplicate face removal."""
        # Create overlapping face detections
        faces = [
            (100, 100, 50, 50),  # Face 1
            (105, 105, 50, 50),  # Overlapping with Face 1
            (200, 200, 60, 60)   # Face 2 (separate)
        ]
        
        unique_faces = face_detector._remove_duplicate_faces(faces, overlap_threshold=0.3)
        
        # Should have 2 unique faces (overlapping ones merged)
        assert len(unique_faces) == 2
    
    def test_calculate_overlap(self, face_detector):
        """Test overlap calculation between two faces."""
        face1 = (100, 100, 50, 50)
        face2 = (110, 110, 50, 50)  # Overlapping
        face3 = (200, 200, 50, 50)  # Not overlapping
        
        overlap1 = face_detector._calculate_overlap(face1, face2)
        overlap2 = face_detector._calculate_overlap(face1, face3)
        
        assert overlap1 > 0  # Should have some overlap
        assert overlap2 == 0  # Should have no overlap
    
    @patch('cv2.dnn.readNetFromCaffe')
    @patch('os.path.exists')
    def test_dnn_detector_initialization(self, mock_exists, mock_read_net, face_detector):
        """Test DNN detector initialization."""
        # Mock file existence and network loading
        mock_exists.return_value = True
        mock_net = Mock()
        mock_read_net.return_value = mock_net
        
        face_detector._initialize_dnn_detector()
        
        assert face_detector.dnn_net is not None
    
    @patch('os.path.exists')
    def test_dnn_detector_initialization_no_files(self, mock_exists, face_detector):
        """Test DNN detector initialization without model files."""
        # Mock files not existing
        mock_exists.return_value = False
        
        face_detector._initialize_dnn_detector()
        
        assert face_detector.dnn_net is None


class TestFaceAnnotator:
    """Test cases for FaceAnnotator class."""
    
    @pytest.fixture
    def face_annotator(self):
        """Create FaceAnnotator instance for testing."""
        return FaceAnnotator()
    
    def test_face_annotator_initialization(self, face_annotator):
        """Test FaceAnnotator initialization."""
        assert 'known' in face_annotator.colors
        assert 'unknown' in face_annotator.colors
        assert 'detected' in face_annotator.colors
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_face_boxes(self, mock_get_text_size, mock_put_text, mock_rectangle, face_annotator):
        """Test drawing face bounding boxes."""
        # Setup mocks
        mock_get_text_size.return_value = ((100, 20), 5)  # (text_size, baseline)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [(100, 100, 50, 50), (200, 200, 60, 60)]
        labels = ["John Doe", "Unknown"]
        face_types = ["known", "unknown"]
        
        annotated_frame = face_annotator.draw_face_boxes(test_frame, faces, labels, face_types)
        
        # Verify rectangle and text drawing were called
        assert mock_rectangle.call_count >= len(faces)  # At least one rectangle per face
        assert mock_put_text.call_count >= len([l for l in labels if l])  # One text per label
    
    def test_draw_face_boxes_no_labels(self, face_annotator):
        """Test drawing face boxes without labels."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [(100, 100, 50, 50)]
        
        with patch('cv2.rectangle') as mock_rectangle:
            annotated_frame = face_annotator.draw_face_boxes(test_frame, faces)
            
            # Should still draw rectangles
            mock_rectangle.assert_called()
    
    @patch('cv2.putText')
    def test_draw_detection_info(self, mock_put_text, face_annotator):
        """Test drawing detection information."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        annotated_frame = face_annotator.draw_detection_info(
            test_frame, 
            num_faces=2, 
            detection_time=0.05
        )
        
        # Should draw face count and detection time
        assert mock_put_text.call_count == 2
    
    def test_draw_detection_info_no_time(self, face_annotator):
        """Test drawing detection info without detection time."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.putText') as mock_put_text:
            annotated_frame = face_annotator.draw_detection_info(test_frame, num_faces=1)
            
            # Should only draw face count
            assert mock_put_text.call_count == 1


# Performance and edge case tests
class TestFaceDetectionPerformance:
    """Performance and edge case tests."""
    
    @pytest.fixture
    def face_detector(self):
        """Create FaceDetector for performance tests."""
        return FaceDetector(method="haar")
    
    def test_large_frame_detection(self, face_detector):
        """Test face detection on large frame."""
        # Create large frame
        large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Should handle large frames without crashing
        faces = face_detector.detect_faces(large_frame)
        assert isinstance(faces, list)
    
    def test_small_frame_detection(self, face_detector):
        """Test face detection on very small frame."""
        # Create small frame
        small_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Should handle small frames
        faces = face_detector.detect_faces(small_frame)
        assert isinstance(faces, list)
    
    def test_grayscale_frame_detection(self, face_detector):
        """Test face detection on grayscale frame."""
        # Create grayscale frame (2D array)
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        
        # Should handle grayscale frames or convert appropriately
        try:
            faces = face_detector.detect_faces(gray_frame)
            assert isinstance(faces, list)
        except Exception as e:
            # It's acceptable if grayscale isn't supported
            assert "color" in str(e).lower() or "bgr" in str(e).lower()


# Integration tests
class TestFaceDetectionIntegration:
    """Integration tests for face detection components."""
    
    @patch('cv2.CascadeClassifier')
    def test_detector_annotator_integration(self, mock_cascade):
        """Test integration between detector and annotator."""
        # Setup detector
        mock_cascade_instance = Mock()
        mock_cascade_instance.empty.return_value = False
        mock_cascade_instance.detectMultiScale.return_value = np.array([[100, 100, 50, 50]])
        mock_cascade.return_value = mock_cascade_instance
        
        detector = FaceDetector(method="haar")
        detector.haar_cascade = mock_cascade_instance
        annotator = FaceAnnotator()
        
        # Test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detect faces
        faces = detector.detect_faces(test_frame)
        
        # Annotate frame
        with patch('cv2.rectangle'), patch('cv2.putText'):
            annotated_frame = annotator.draw_face_boxes(
                test_frame, faces, ["Test Person"], ["known"]
            )
        
        assert len(faces) == 1
        assert annotated_frame.shape == test_frame.shape


if __name__ == "__main__":
    pytest.main([__file__])
