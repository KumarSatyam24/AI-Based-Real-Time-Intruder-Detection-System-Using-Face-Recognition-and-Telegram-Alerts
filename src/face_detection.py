"""
Face Detection Module
Epic 3: Face Detection Module - Story Points 7, 8

Implements face detection using OpenCV Haar Cascades and DNN-based detectors.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger
import os


class FaceDetector:
    """
    Face detection class supporting multiple detection methods.
    """
    
    def __init__(self, method: str = "haar", confidence_threshold: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            method: Detection method - "haar", "dnn", or "both"
            confidence_threshold: Confidence threshold for DNN detection
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        # Haar Cascade classifier
        self.haar_cascade = None
        
        # DNN face detector
        self.dnn_net = None
        
        # Initialize detectors
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize the selected face detectors."""
        
        if self.method in ["haar", "both"]:
            self._initialize_haar_cascade()
        
        if self.method in ["dnn", "both"]:
            self._initialize_dnn_detector()
    
    def _initialize_haar_cascade(self):
        """Initialize Haar Cascade face detector."""
        try:
            # Load pre-trained Haar cascade for frontal faces
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            if not os.path.exists(cascade_path):
                logger.error(f"Haar cascade file not found: {cascade_path}")
                return
            
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.haar_cascade.empty():
                logger.error("Failed to load Haar cascade classifier")
                self.haar_cascade = None
            else:
                logger.info("Haar cascade face detector initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing Haar cascade: {e}")
            self.haar_cascade = None
    
    def _initialize_dnn_detector(self):
        """Initialize DNN-based face detector."""
        try:
            # Download DNN model files if not present
            model_dir = "models"
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            # Create models directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if model files exist, if not, provide instructions
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                logger.warning("DNN model files not found. Please download them manually:")
                logger.warning("1. deploy.prototxt")
                logger.warning("2. res10_300x300_ssd_iter_140000.caffemodel")
                logger.warning("Place them in the 'models' directory")
                return
            
            # Load DNN model
            self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("DNN face detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DNN detector: {e}")
            self.dnn_net = None
    
    def detect_faces_haar(self, frame: np.ndarray, scale_factor: float = 1.1, 
                         min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade classifier.
        
        Args:
            frame: Input frame
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should retain
            min_size: Minimum possible face size
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.haar_cascade is None:
            return []
        
        try:
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of tuples
            face_locations = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            
            logger.debug(f"Haar cascade detected {len(face_locations)} faces")
            return face_locations
            
        except Exception as e:
            logger.error(f"Error in Haar cascade face detection: {e}")
            return []
    
    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN-based detector.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.dnn_net is None:
            return []
        
        try:
            h, w = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            
            # Set blob as input to network
            self.dnn_net.setInput(blob)
            
            # Run forward pass
            detections = self.dnn_net.forward()
            
            face_locations = []
            
            # Process detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    
                    # Convert to (x, y, w, h) format
                    width = x1 - x
                    height = y1 - y
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 0 and height > 0:
                        face_locations.append((x, y, width, height))
            
            logger.debug(f"DNN detector found {len(face_locations)} faces")
            return face_locations
            
        except Exception as e:
            logger.error(f"Error in DNN face detection: {e}")
            return []
    
    def detect_faces(self, frame: np.ndarray, **kwargs) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using the configured method.
        
        Args:
            frame: Input frame
            **kwargs: Additional parameters for detection methods
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if frame is None or frame.size == 0:
            return []
        
        faces = []
        
        if self.method == "haar":
            faces = self.detect_faces_haar(frame, **kwargs)
        elif self.method == "dnn":
            faces = self.detect_faces_dnn(frame)
        elif self.method == "both":
            # Use both methods and combine results
            haar_faces = self.detect_faces_haar(frame, **kwargs)
            dnn_faces = self.detect_faces_dnn(frame)
            
            # Combine and remove duplicates (simple approach)
            all_faces = haar_faces + dnn_faces
            faces = self._remove_duplicate_faces(all_faces)
        
        return faces
    
    def _remove_duplicate_faces(self, faces: List[Tuple[int, int, int, int]], 
                               overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        Remove duplicate face detections based on overlap.
        
        Args:
            faces: List of face bounding boxes
            overlap_threshold: Minimum overlap ratio to consider faces as duplicates
            
        Returns:
            List of unique face bounding boxes
        """
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            
            for unique_face in unique_faces:
                overlap = self._calculate_overlap(face, unique_face)
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_overlap(self, face1: Tuple[int, int, int, int], 
                          face2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio between two bounding boxes.
        
        Args:
            face1: First bounding box (x, y, w, h)
            face2: Second bounding box (x, y, w, h)
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area


class FaceAnnotator:
    """
    Helper class for drawing face detection annotations on frames.
    """
    
    def __init__(self):
        """Initialize face annotator."""
        self.colors = {
            'known': (0, 255, 0),      # Green for known faces
            'unknown': (0, 0, 255),    # Red for unknown faces
            'detected': (255, 0, 0)    # Blue for detected faces (no recognition)
        }
    
    def draw_face_boxes(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                       labels: Optional[List[str]] = None, 
                       face_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on detected faces.
        
        Args:
            frame: Input frame
            faces: List of face bounding boxes (x, y, w, h)
            labels: Optional list of face labels/names
            face_types: Optional list of face types ('known', 'unknown', 'detected')
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Determine color based on face type
            face_type = face_types[i] if face_types and i < len(face_types) else 'detected'
            color = self.colors.get(face_type, self.colors['detected'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label if provided
            if labels and i < len(labels):
                label = labels[i]
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(annotated_frame, 
                             (x, y - text_height - 10), 
                             (x + text_width, y), 
                             color, -1)
                
                # Draw text
                cv2.putText(annotated_frame, label, 
                           (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_detection_info(self, frame: np.ndarray, num_faces: int, 
                           detection_time: Optional[float] = None) -> np.ndarray:
        """
        Draw detection information on frame.
        
        Args:
            frame: Input frame
            num_faces: Number of detected faces
            detection_time: Time taken for detection (in seconds)
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw face count
        info_text = f"Faces: {num_faces}"
        cv2.putText(annotated_frame, info_text, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        
        # Draw detection time if provided
        if detection_time is not None:
            time_text = f"Detection: {detection_time*1000:.1f}ms"
            cv2.putText(annotated_frame, time_text, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
        
        return annotated_frame
