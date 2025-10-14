"""Interactive ROI selector for damage detection zones."""
from __future__ import annotations

import json
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


class ROISelector:
    """Interactive tool to select door/window ROIs by clicking and dragging."""
    
    def __init__(self, frame: np.ndarray, video_name: str):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.video_name = video_name
        self.rois: List[Tuple[str, Tuple[int, int, int, int]]] = []
        self.current_roi: Optional[List[int]] = None
        self.mode = 'door'  # 'door' or 'window'
        self.drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_roi:
                self.current_roi[2] = x
                self.current_roi[3] = y
                self._update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_roi:
                self.drawing = False
                self.current_roi[2] = x
                self.current_roi[3] = y
                # Convert to (x, y, w, h)
                x1, y1, x2, y2 = self.current_roi
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                w, h = x_max - x_min, y_max - y_min
                if w > 10 and h > 10:  # minimum size
                    self.rois.append((self.mode, (x_min, y_min, w, h)))
                    print(f"Added {self.mode} ROI: ({x_min}, {y_min}, {w}, {h})")
                self.current_roi = None
                self._update_display()
    
    def _update_display(self):
        self.display_frame = self.frame.copy()
        # Draw existing ROIs
        for roi_type, (x, y, w, h) in self.rois:
            color = (0, 0, 255) if roi_type == 'door' else (255, 0, 0)
            cv2.rectangle(self.display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.display_frame, roi_type, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw current ROI being drawn
        if self.current_roi:
            x1, y1, x2, y2 = self.current_roi
            color = (0, 255, 255)
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Instructions
        mode_color = (0, 0, 255) if self.mode == 'door' else (255, 0, 0)
        cv2.putText(self.display_frame, f"Mode: {self.mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(self.display_frame, "D=door W=window S=save Q=quit U=undo", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.display_frame, f"ROIs: {len(self.rois)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ROI Selector', self.display_frame)
    
    def run(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Run interactive selector. Returns list of (roi_type, (x,y,w,h))."""
        cv2.namedWindow('ROI Selector')
        cv2.setMouseCallback('ROI Selector', self.mouse_callback)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d') or key == ord('D'):
                self.mode = 'door'
                self._update_display()
                print("Mode: DOOR")
                
            elif key == ord('w') or key == ord('W'):
                self.mode = 'window'
                self._update_display()
                print("Mode: WINDOW")
                
            elif key == ord('u') or key == ord('U'):
                if self.rois:
                    removed = self.rois.pop()
                    print(f"Removed {removed[0]} ROI")
                    self._update_display()
                    
            elif key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                return self.rois
                
            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return []
        
        cv2.destroyAllWindows()
        return self.rois


def save_rois_to_file(video_name: str, rois: List[Tuple[str, Tuple[int, int, int, int]]], 
                      roi_dir: str = './data/rois/') -> str:
    """Save ROIs to JSON file named after the video."""
    os.makedirs(roi_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    roi_file = os.path.join(roi_dir, f"{base_name}_rois.json")
    
    data = {
        'video': video_name,
        'doors': [],
        'windows': []
    }
    
    for roi_type, coords in rois:
        if roi_type == 'door':
            data['doors'].append(list(coords))
        else:
            data['windows'].append(list(coords))
    
    with open(roi_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved ROIs to: {roi_file}")
    return roi_file


def load_rois_from_file(video_name: str, roi_dir: str = './data/rois/') -> Tuple[List, List]:
    """Load ROIs from JSON file. Returns (door_rois, window_rois)."""
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    roi_file = os.path.join(roi_dir, f"{base_name}_rois.json")
    
    if not os.path.exists(roi_file):
        return [], []
    
    try:
        with open(roi_file, 'r') as f:
            data = json.load(f)
        
        doors = [tuple(roi) for roi in data.get('doors', [])]
        windows = [tuple(roi) for roi in data.get('windows', [])]
        
        print(f"Loaded ROIs from {roi_file}: {len(doors)} doors, {len(windows)} windows")
        return doors, windows
    except Exception as e:
        print(f"Failed to load ROIs: {e}")
        return [], []


def select_rois_for_video(video_path: str, frame_width: int = 640, frame_height: int = 480) -> bool:
    """Open video, let user select ROIs, and save them. Returns True if saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame from video")
        return False
    
    # Resize to match processing size
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    print(f"\nROI Selection for: {os.path.basename(video_path)}")
    print("Instructions:")
    print("  - Click and drag to draw rectangles")
    print("  - Press D for door mode, W for window mode")
    print("  - Press U to undo last ROI")
    print("  - Press S to save and exit")
    print("  - Press Q to quit without saving")
    
    selector = ROISelector(frame, video_path)
    rois = selector.run()
    
    if rois:
        save_rois_to_file(video_path, rois)
        print(f"Saved {len(rois)} ROI(s)")
        return True
    else:
        print("No ROIs saved")
        return False
