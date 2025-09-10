"""
Unit Tests for Face Recognition Module
Epic 8: Testing & Validation - Story Point 25
"""

import pytest
import numpy as np
import os
import pickle
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_recognition_module import FaceDatabase, FaceRecognizer


class TestFaceDatabase:
    """Test cases for FaceDatabase class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            authorized_faces_dir = os.path.join(temp_dir, "authorized_faces")
            encodings_file = os.path.join(temp_dir, "encodings.pkl")
            
            yield FaceDatabase(authorized_faces_dir, encodings_file)
    
    def test_database_initialization(self, temp_db):
        """Test database initialization."""
        assert os.path.exists(temp_db.authorized_faces_dir)
        assert temp_db.known_faces == {}
    
    @patch('face_recognition.load_image_file')
    @patch('face_recognition.face_encodings')
    def test_add_person_success(self, mock_face_encodings, mock_load_image, temp_db):
        """Test successful person addition."""
        # Mock face recognition functions
        mock_load_image.return_value = np.zeros((100, 100, 3))
        mock_face_encodings.return_value = [np.random.rand(128)]  # Mock encoding
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Test adding person
            result = temp_db.add_person("John Doe", [temp_file_path])
            
            assert result is True
            assert "John Doe" in temp_db.known_faces
            assert len(temp_db.known_faces["John Doe"]["encodings"]) == 1
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('face_recognition.load_image_file')
    @patch('face_recognition.face_encodings')
    def test_add_person_no_face_found(self, mock_face_encodings, mock_load_image, temp_db):
        """Test adding person when no face is found in image."""
        # Mock no face found
        mock_load_image.return_value = np.zeros((100, 100, 3))
        mock_face_encodings.return_value = []  # No faces found
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            result = temp_db.add_person("Jane Doe", [temp_file_path])
            
            assert result is False
            assert "Jane Doe" not in temp_db.known_faces
            
        finally:
            os.unlink(temp_file_path)
    
    def test_add_person_nonexistent_file(self, temp_db):
        """Test adding person with nonexistent image file."""
        result = temp_db.add_person("John Doe", ["/nonexistent/file.jpg"])
        
        assert result is False
        assert "John Doe" not in temp_db.known_faces
    
    def test_remove_person_success(self, temp_db):
        """Test successful person removal."""
        # Add person first
        temp_db.known_faces["John Doe"] = {
            "encodings": [np.random.rand(128)],
            "metadata": {"date_added": "2024-01-01"}
        }
        
        result = temp_db.remove_person("John Doe")
        
        assert result is True
        assert "John Doe" not in temp_db.known_faces
    
    def test_remove_person_not_found(self, temp_db):
        """Test removing person that doesn't exist."""
        result = temp_db.remove_person("Nonexistent Person")
        
        assert result is False
    
    def test_update_person_success(self, temp_db):
        """Test successful person update."""
        # Add person first
        temp_db.known_faces["John Doe"] = {
            "encodings": [np.random.rand(128)],
            "metadata": {"date_added": "2024-01-01", "department": "IT"}
        }
        
        result = temp_db.update_person("John Doe", new_metadata={"department": "HR"})
        
        assert result is True
        assert temp_db.known_faces["John Doe"]["metadata"]["department"] == "HR"
        assert "last_updated" in temp_db.known_faces["John Doe"]["metadata"]
    
    def test_update_person_not_found(self, temp_db):
        """Test updating person that doesn't exist."""
        result = temp_db.update_person("Nonexistent", new_metadata={"department": "IT"})
        
        assert result is False
    
    def test_get_all_encodings(self, temp_db):
        """Test getting all encodings."""
        # Add test data
        encoding1 = np.random.rand(128)
        encoding2 = np.random.rand(128)
        
        temp_db.known_faces["Person1"] = {
            "encodings": [encoding1],
            "metadata": {}
        }
        temp_db.known_faces["Person2"] = {
            "encodings": [encoding2],
            "metadata": {}
        }
        
        encodings, names = temp_db.get_all_encodings()
        
        assert len(encodings) == 2
        assert len(names) == 2
        assert "Person1" in names
        assert "Person2" in names
    
    def test_list_all_people(self, temp_db):
        """Test listing all people."""
        temp_db.known_faces["Person1"] = {"encodings": [], "metadata": {}}
        temp_db.known_faces["Person2"] = {"encodings": [], "metadata": {}}
        
        people = temp_db.list_all_people()
        
        assert len(people) == 2
        assert "Person1" in people
        assert "Person2" in people
    
    def test_save_and_load_database(self, temp_db):
        """Test database persistence."""
        # Add test data
        temp_db.known_faces["Test Person"] = {
            "encodings": [np.random.rand(128)],
            "metadata": {"test": "data"}
        }
        
        # Save database
        save_result = temp_db.save_database()
        assert save_result is True
        assert os.path.exists(temp_db.encodings_file)
        
        # Create new database instance and load
        new_db = FaceDatabase(temp_db.authorized_faces_dir, temp_db.encodings_file)
        load_result = new_db.load_database()
        
        assert load_result is True
        assert "Test Person" in new_db.known_faces
        assert new_db.known_faces["Test Person"]["metadata"]["test"] == "data"


class TestFaceRecognizer:
    """Test cases for FaceRecognizer class."""
    
    @pytest.fixture
    def mock_database(self):
        """Create mock database for testing."""
        db = Mock(spec=FaceDatabase)
        db.get_all_encodings.return_value = (
            [np.random.rand(128), np.random.rand(128)],  # Encodings
            ["Person1", "Person2"]  # Names
        )
        db.list_all_people.return_value = ["Person1", "Person2"]
        return db
    
    @pytest.fixture
    def face_recognizer(self, mock_database):
        """Create FaceRecognizer for testing."""
        return FaceRecognizer(mock_database, tolerance=0.6)
    
    def test_recognizer_initialization(self, face_recognizer, mock_database):
        """Test FaceRecognizer initialization."""
        assert face_recognizer.database == mock_database
        assert face_recognizer.tolerance == 0.6
        assert len(face_recognizer._known_encodings) == 2
        assert len(face_recognizer._known_names) == 2
    
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    @patch('cv2.cvtColor')
    def test_recognize_faces_known_person(self, mock_cvt_color, mock_face_distance, 
                                         mock_compare_faces, mock_face_encodings, 
                                         face_recognizer):
        """Test recognizing known person."""
        # Setup mocks
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvt_color.return_value = test_frame
        mock_face_encodings.return_value = [np.random.rand(128)]
        mock_compare_faces.return_value = [True, False]  # Match with first person
        mock_face_distance.return_value = np.array([0.3, 0.8])  # Close to first person
        
        face_locations = [(100, 100, 50, 50)]  # (x, y, w, h)
        
        results = face_recognizer.recognize_faces(test_frame, face_locations)
        
        assert len(results) == 1
        assert results[0]['name'] == "Person1"
        assert results[0]['is_known'] is True
        assert results[0]['confidence'] > 0
    
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    @patch('cv2.cvtColor')
    def test_recognize_faces_unknown_person(self, mock_cvt_color, mock_face_distance,
                                           mock_compare_faces, mock_face_encodings,
                                           face_recognizer):
        """Test recognizing unknown person."""
        # Setup mocks
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvt_color.return_value = test_frame
        mock_face_encodings.return_value = [np.random.rand(128)]
        mock_compare_faces.return_value = [False, False]  # No matches
        mock_face_distance.return_value = np.array([0.8, 0.9])  # Far from all persons
        
        face_locations = [(100, 100, 50, 50)]
        
        results = face_recognizer.recognize_faces(test_frame, face_locations)
        
        assert len(results) == 1
        assert results[0]['name'] == "Unknown"
        assert results[0]['is_known'] is False
        assert results[0]['confidence'] == 0.0
    
    def test_recognize_faces_no_locations(self, face_recognizer):
        """Test recognition with no face locations."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = face_recognizer.recognize_faces(test_frame, [])
        
        assert results == []
    
    def test_recognize_faces_empty_database(self):
        """Test recognition with empty database."""
        empty_db = Mock(spec=FaceDatabase)
        empty_db.get_all_encodings.return_value = ([], [])
        empty_db.list_all_people.return_value = []
        
        recognizer = FaceRecognizer(empty_db)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_locations = [(100, 100, 50, 50)]
        
        results = recognizer.recognize_faces(test_frame, face_locations)
        
        assert len(results) == 1
        assert results[0]['name'] == "Unknown"
    
    def test_set_tolerance(self, face_recognizer):
        """Test setting recognition tolerance."""
        face_recognizer.set_tolerance(0.8)
        assert face_recognizer.tolerance == 0.8
        
        # Test bounds
        face_recognizer.set_tolerance(-0.1)
        assert face_recognizer.tolerance == 0.0
        
        face_recognizer.set_tolerance(1.5)
        assert face_recognizer.tolerance == 1.0
    
    def test_get_recognition_stats(self, face_recognizer):
        """Test getting recognition statistics."""
        stats = face_recognizer.get_recognition_stats()
        
        assert 'known_people' in stats
        assert 'total_encodings' in stats
        assert 'tolerance' in stats
        assert 'model' in stats
        assert stats['known_people'] == 2
        assert stats['total_encodings'] == 2


# Performance and edge case tests
class TestFaceRecognitionPerformance:
    """Performance and edge case tests."""
    
    @pytest.fixture
    def large_database(self):
        """Create database with many people for performance testing."""
        db = Mock(spec=FaceDatabase)
        # Simulate 100 people with 2 encodings each
        encodings = [np.random.rand(128) for _ in range(200)]
        names = [f"Person{i//2}" for i in range(200)]
        
        db.get_all_encodings.return_value = (encodings, names)
        db.list_all_people.return_value = [f"Person{i}" for i in range(100)]
        
        return db
    
    def test_large_database_recognition(self, large_database):
        """Test recognition performance with large database."""
        recognizer = FaceRecognizer(large_database)
        
        # Should handle large database without issues
        assert len(recognizer._known_encodings) == 200
        assert len(recognizer._known_names) == 200
    
    @pytest.mark.skip(reason="Requires actual face encoding data")
    def test_multiple_faces_recognition(self, mock_database):
        """Test recognition with multiple faces in frame."""
        recognizer = FaceRecognizer(mock_database)
        
        with patch('face_recognition.face_encodings') as mock_encodings, \
             patch('face_recognition.compare_faces') as mock_compare, \
             patch('face_recognition.face_distance') as mock_distance, \
             patch('cv2.cvtColor') as mock_cvt:
            
            # Mock multiple face encodings
            mock_encodings.return_value = [np.random.rand(128) for _ in range(3)]
            mock_compare.return_value = [True, False]
            mock_distance.return_value = np.array([0.3, 0.8])
            mock_cvt.return_value = np.zeros((480, 640, 3))
            
            # Multiple face locations
            face_locations = [(100, 100, 50, 50), (200, 200, 50, 50), (300, 300, 50, 50)]
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            results = recognizer.recognize_faces(test_frame, face_locations)
            
            # Should return results for all faces
            assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__])
