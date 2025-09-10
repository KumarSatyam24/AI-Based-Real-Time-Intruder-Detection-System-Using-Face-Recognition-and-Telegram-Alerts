"""
Face Recognition Module
Epic 4: Face Recognition Module - Story Points 10, 11, 12, 13

Implements face recognition using face embeddings and similarity comparison.
"""

import cv2
import numpy as np
import face_recognition
import pickle
import os
from typing import List, Tuple, Dict, Optional, Any
from loguru import logger
from datetime import datetime
import json


class FaceDatabase:
    """
    Class for managing the face database with authorized personnel.
    """
    
    def __init__(self, authorized_faces_dir: str, encodings_file: str):
        """
        Initialize face database.
        
        Args:
            authorized_faces_dir: Directory containing authorized face images
            encodings_file: Path to pickle file storing face encodings
        """
        self.authorized_faces_dir = authorized_faces_dir
        self.encodings_file = encodings_file
        self.known_faces = {}  # {name: {"encodings": [...], "metadata": {...}}}
        
        # Create directories if they don't exist
        os.makedirs(authorized_faces_dir, exist_ok=True)
        os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
        
        # Load existing database
        self.load_database()
    
    def add_person(self, name: str, image_paths: List[str], 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new person to the database.
        
        Args:
            name: Name of the person
            image_paths: List of paths to face images
            metadata: Optional metadata (date_added, department, etc.)
            
        Returns:
            bool: True if person added successfully, False otherwise
        """
        try:
            encodings = []
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Load and encode face
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) == 0:
                    logger.warning(f"No face found in image: {image_path}")
                    continue
                
                if len(face_encodings) > 1:
                    logger.warning(f"Multiple faces found in image: {image_path}. Using first one.")
                
                encodings.append(face_encodings[0])
            
            if not encodings:
                logger.error(f"No valid face encodings found for {name}")
                return False
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'date_added': datetime.now().isoformat(),
                'num_images': len(encodings),
                'image_paths': image_paths
            })
            
            # Add to database
            self.known_faces[name] = {
                'encodings': encodings,
                'metadata': metadata
            }
            
            logger.info(f"Added {name} to database with {len(encodings)} encodings")
            
            # Save database
            self.save_database()
            return True
            
        except Exception as e:
            logger.error(f"Error adding person {name}: {e}")
            return False
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person from the database.
        
        Args:
            name: Name of the person to remove
            
        Returns:
            bool: True if person removed successfully, False otherwise
        """
        if name in self.known_faces:
            del self.known_faces[name]
            logger.info(f"Removed {name} from database")
            self.save_database()
            return True
        else:
            logger.warning(f"Person {name} not found in database")
            return False
    
    def update_person(self, name: str, new_image_paths: List[str] = None, 
                     new_metadata: Dict[str, Any] = None) -> bool:
        """
        Update a person's information in the database.
        
        Args:
            name: Name of the person to update
            new_image_paths: New face images to add
            new_metadata: New metadata to update
            
        Returns:
            bool: True if person updated successfully, False otherwise
        """
        if name not in self.known_faces:
            logger.warning(f"Person {name} not found in database")
            return False
        
        try:
            # Update encodings if new images provided
            if new_image_paths:
                additional_encodings = []
                
                for image_path in new_image_paths:
                    if not os.path.exists(image_path):
                        logger.warning(f"Image not found: {image_path}")
                        continue
                    
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        additional_encodings.append(face_encodings[0])
                
                if additional_encodings:
                    self.known_faces[name]['encodings'].extend(additional_encodings)
                    
                    # Update metadata
                    current_paths = self.known_faces[name]['metadata'].get('image_paths', [])
                    current_paths.extend(new_image_paths)
                    self.known_faces[name]['metadata']['image_paths'] = current_paths
                    self.known_faces[name]['metadata']['num_images'] = len(self.known_faces[name]['encodings'])
                    self.known_faces[name]['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Update metadata if provided
            if new_metadata:
                self.known_faces[name]['metadata'].update(new_metadata)
                self.known_faces[name]['metadata']['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Updated {name} in database")
            self.save_database()
            return True
            
        except Exception as e:
            logger.error(f"Error updating person {name}: {e}")
            return False
    
    def get_all_encodings(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get all face encodings and corresponding names.
        
        Returns:
            Tuple of (encodings_list, names_list)
        """
        all_encodings = []
        all_names = []
        
        for name, data in self.known_faces.items():
            for encoding in data['encodings']:
                all_encodings.append(encoding)
                all_names.append(name)
        
        return all_encodings, all_names
    
    def get_person_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific person.
        
        Args:
            name: Name of the person
            
        Returns:
            Dictionary containing person's information or None if not found
        """
        return self.known_faces.get(name)
    
    def list_all_people(self) -> List[str]:
        """
        Get list of all people in the database.
        
        Returns:
            List of names
        """
        return list(self.known_faces.keys())
    
    def save_database(self) -> bool:
        """
        Save database to file.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.debug("Face database saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False
    
    def load_database(self) -> bool:
        """
        Load database from file.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.encodings_file):
            logger.info("No existing database found. Starting with empty database.")
            return True
        
        try:
            with open(self.encodings_file, 'rb') as f:
                self.known_faces = pickle.load(f)
            
            num_people = len(self.known_faces)
            total_encodings = sum(len(data['encodings']) for data in self.known_faces.values())
            
            logger.info(f"Loaded database: {num_people} people, {total_encodings} total encodings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.known_faces = {}
            return False


class FaceRecognizer:
    """
    Face recognition class using face encodings and similarity comparison.
    """
    
    def __init__(self, database: FaceDatabase, tolerance: float = 0.6, 
                 model: str = "small"):
        """
        Initialize face recognizer.
        
        Args:
            database: FaceDatabase instance
            tolerance: Face distance tolerance for recognition
            model: Face recognition model ("small" for speed, "large" for accuracy)
        """
        self.database = database
        self.tolerance = tolerance
        self.model = model
        
        # Cache for known encodings (for performance)
        self._known_encodings = []
        self._known_names = []
        self._last_database_update = None
        
        self._update_encoding_cache()
    
    def _update_encoding_cache(self):
        """Update the encoding cache from database."""
        self._known_encodings, self._known_names = self.database.get_all_encodings()
        self._last_database_update = datetime.now()
        logger.debug(f"Updated encoding cache: {len(self._known_encodings)} encodings")
    
    def recognize_faces(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
        """
        Recognize faces in the given frame.
        
        Args:
            frame: Input frame (BGR format)
            face_locations: List of face bounding boxes (x, y, w, h)
            
        Returns:
            List of dictionaries containing recognition results
        """
        if not face_locations:
            return []
        
        # Update cache if database was modified
        current_people = self.database.list_all_people()
        if len(current_people) != len(set(self._known_names)):
            self._update_encoding_cache()
        
        if not self._known_encodings:
            logger.warning("No known faces in database")
            return [{'name': 'Unknown', 'confidence': 0.0, 'distance': 1.0} for _ in face_locations]
        
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert (x, y, w, h) to (top, right, bottom, left) format
            face_locations_dlib = []
            for (x, y, w, h) in face_locations:
                top = y
                right = x + w
                bottom = y + h
                left = x
                face_locations_dlib.append((top, right, bottom, left))
            
            # Get face encodings for detected faces
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations_dlib,
                model=self.model
            )
            
            results = []
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self._known_encodings, 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                
                # Calculate distances
                face_distances = face_recognition.face_distance(
                    self._known_encodings, 
                    face_encoding
                )
                
                name = "Unknown"
                confidence = 0.0
                min_distance = 1.0
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    if matches[best_match_index]:
                        name = self._known_names[best_match_index]
                        # Convert distance to confidence (0-1 scale)
                        confidence = max(0.0, 1.0 - min_distance)
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'distance': min_distance,
                    'is_known': name != "Unknown"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return [{'name': 'Error', 'confidence': 0.0, 'distance': 1.0} for _ in face_locations]
    
    def set_tolerance(self, tolerance: float):
        """
        Set face recognition tolerance.
        
        Args:
            tolerance: New tolerance value (0.0 to 1.0)
        """
        self.tolerance = max(0.0, min(1.0, tolerance))
        logger.info(f"Face recognition tolerance set to {self.tolerance}")
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """
        Get recognition statistics.
        
        Returns:
            Dictionary containing recognition statistics
        """
        return {
            'known_people': len(self.database.list_all_people()),
            'total_encodings': len(self._known_encodings),
            'tolerance': self.tolerance,
            'model': self.model,
            'last_cache_update': self._last_database_update.isoformat() if self._last_database_update else None
        }


class RecognitionResult:
    """
    Class to represent face recognition results with additional metadata.
    """
    
    def __init__(self, name: str, confidence: float, distance: float, 
                 bounding_box: Tuple[int, int, int, int], timestamp: Optional[datetime] = None):
        """
        Initialize recognition result.
        
        Args:
            name: Recognized name or "Unknown"
            confidence: Recognition confidence (0.0 to 1.0)
            distance: Face distance from known faces
            bounding_box: Face bounding box (x, y, w, h)
            timestamp: Detection timestamp
        """
        self.name = name
        self.confidence = confidence
        self.distance = distance
        self.bounding_box = bounding_box
        self.timestamp = timestamp or datetime.now()
        self.is_known = name != "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'name': self.name,
            'confidence': self.confidence,
            'distance': self.distance,
            'bounding_box': self.bounding_box,
            'timestamp': self.timestamp.isoformat(),
            'is_known': self.is_known
        }
    
    def __repr__(self) -> str:
        return f"RecognitionResult(name='{self.name}', confidence={self.confidence:.2f}, is_known={self.is_known})"
