"""Incident logging and reporting system."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

from .config import Config, get_logger
from .damage import DamageEvent


@dataclass
class MotionIncident:
    """Record of a motion/intrusion detection event."""
    timestamp: str
    frame_number: int
    video_file: str
    event_type: str  # 'motion_detected', 'person_detected', 'intrusion'
    person_count: int
    bounding_boxes: List[List[int]]  # List of [x, y, w, h]
    confidence: Optional[float] = None


@dataclass
class DamageIncident:
    """Record of a damage detection event."""
    timestamp: str
    frame_number: int
    video_file: str
    roi_type: str  # 'door' or 'window'
    roi_index: int
    event_type: str  # 'forced_entry' or 'shatter'
    score: float
    roi_coordinates: List[int]  # [x, y, w, h]


@dataclass
class VideoProcessingSummary:
    """Summary of entire video processing session."""
    video_file: str
    start_time: str
    end_time: str
    total_frames: int
    motion_incidents: int
    damage_incidents: int
    output_video: str
    configuration: dict


class IncidentRecorder:
    """Records and manages incident logs during video processing."""

    def __init__(self, cfg: Config, video_file: str):
        self.cfg = cfg
        self.video_file = os.path.basename(video_file)
        self.logger = get_logger()
        
        # Session tracking
        self.session_start = datetime.now()
        self.motion_incidents: List[MotionIncident] = []
        self.damage_incidents: List[DamageIncident] = []
        self.total_frames = 0
        self.output_video_path: Optional[str] = None
        
        # Create incidents directory
        os.makedirs(cfg.INCIDENT_LOG_DIR, exist_ok=True)
        
    def log_motion_event(
        self,
        frame_number: int,
        event_type: str = 'motion_detected',
        person_count: int = 0,
        bounding_boxes: List[List[int]] = None,
        confidence: Optional[float] = None
    ):
        """Log a motion/intrusion detection event."""
        if not self.cfg.ENABLE_INCIDENT_LOGGING or not self.cfg.LOG_MOTION_EVENTS:
            return
            
        incident = MotionIncident(
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            video_file=self.video_file,
            event_type=event_type,
            person_count=person_count,
            bounding_boxes=bounding_boxes or [],
            confidence=confidence
        )
        self.motion_incidents.append(incident)
        
    def log_damage_event(self, frame_number: int, event: DamageEvent):
        """Log a damage detection event."""
        if not self.cfg.ENABLE_INCIDENT_LOGGING or not self.cfg.LOG_DAMAGE_EVENTS:
            return
            
        # Get ROI coordinates
        rois = self.cfg.DOOR_ROIS if event.roi_type == 'door' else self.cfg.WINDOW_ROIS
        roi_coords = list(rois[event.roi_index]) if event.roi_index < len(rois) else [0, 0, 0, 0]
        
        incident = DamageIncident(
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            video_file=self.video_file,
            roi_type=event.roi_type,
            roi_index=event.roi_index,
            event_type=event.event_type,
            score=event.score,
            roi_coordinates=roi_coords
        )
        self.damage_incidents.append(incident)
        self.logger.info(
            "Damage event logged: %s %s at frame %d (score=%.1f)",
            event.roi_type, event.event_type, frame_number, event.score
        )
        
    def finalize(self, output_video_path: str, total_frames: int) -> str:
        """Finalize and save incident report."""
        self.output_video_path = output_video_path
        self.total_frames = total_frames
        
        if not self.cfg.ENABLE_INCIDENT_LOGGING:
            return None
            
        session_end = datetime.now()
        
        # Create summary
        summary = VideoProcessingSummary(
            video_file=self.video_file,
            start_time=self.session_start.isoformat(),
            end_time=session_end.isoformat(),
            total_frames=total_frames,
            motion_incidents=len(self.motion_incidents),
            damage_incidents=len(self.damage_incidents),
            output_video=os.path.basename(output_video_path) if output_video_path else "N/A",
            configuration={
                'person_only_mode': self.cfg.PERSON_ONLY_MODE,
                'damage_detection_enabled': self.cfg.ENABLE_DAMAGE_DETECTION,
                'damage_diff_threshold': self.cfg.DAMAGE_DIFF_THRESHOLD,
                'damage_edge_spike_ratio': self.cfg.DAMAGE_EDGE_SPIKE_RATIO,
                'damage_fragment_threshold': self.cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD,
                'damage_persistence': self.cfg.DAMAGE_MIN_PERSISTENCE,
            }
        )
        
        # Create compact motion summary (frame ranges instead of every frame)
        motion_summary = self._create_motion_summary()
        
        # Generate CONCISE report
        report = {
            'summary': asdict(summary),
            'motion_summary': motion_summary,
            'damage_incidents': [asdict(d) for d in self.damage_incidents],
            'statistics': {
                'processing_duration_seconds': (session_end - self.session_start).total_seconds(),
                'motion_detection_rate': len(self.motion_incidents) / max(1, total_frames),
                'total_motion_frames': len(self.motion_incidents),
                'total_damage_events': len(self.damage_incidents),
                'unique_damage_rois': len(set((d.roi_type, d.roi_index) for d in self.damage_incidents)),
                'first_intrusion_frame': self.motion_incidents[0].frame_number if self.motion_incidents else None,
                'last_intrusion_frame': self.motion_incidents[-1].frame_number if self.motion_incidents else None,
            }
        }
        
        # Save concise report
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self.video_file)[0]
        output_file = os.path.join(
            self.cfg.INCIDENT_LOG_DIR,
            f"{base_name}_incidents_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            "Incident report saved: %s (%d motion, %d damage events)",
            output_file,
            len(self.motion_incidents),
            len(self.damage_incidents)
        )
        
        return output_file
    
    def _create_motion_summary(self) -> dict:
        """Create a compact summary of motion events using frame ranges."""
        if not self.motion_incidents:
            return {
                'intrusion_periods': [],
                'total_motion_frames': 0,
                'max_consecutive_frames': 0
            }
        
        # Group consecutive frames into periods
        periods = []
        current_period_start = self.motion_incidents[0].frame_number
        current_period_end = current_period_start
        current_person_count = self.motion_incidents[0].person_count
        max_persons = current_person_count
        
        for i in range(1, len(self.motion_incidents)):
            frame_num = self.motion_incidents[i].frame_number
            person_count = self.motion_incidents[i].person_count
            
            # If frames are consecutive (within 5 frames gap), extend period
            if frame_num - current_period_end <= 5:
                current_period_end = frame_num
                max_persons = max(max_persons, person_count)
            else:
                # Save previous period
                periods.append({
                    'start_frame': current_period_start,
                    'end_frame': current_period_end,
                    'duration_frames': current_period_end - current_period_start + 1,
                    'max_persons_detected': max_persons
                })
                # Start new period
                current_period_start = frame_num
                current_period_end = frame_num
                max_persons = person_count
        
        # Add last period
        periods.append({
            'start_frame': current_period_start,
            'end_frame': current_period_end,
            'duration_frames': current_period_end - current_period_start + 1,
            'max_persons_detected': max_persons
        })
        
        max_consecutive = max(p['duration_frames'] for p in periods) if periods else 0
        
        return {
            'intrusion_periods': periods,
            'total_intrusion_periods': len(periods),
            'total_motion_frames': len(self.motion_incidents),
            'max_consecutive_frames': max_consecutive
        }


def generate_summary_report(cfg: Config, incident_files: List[str]) -> str:
    """Generate a consolidated summary report from multiple incident logs."""
    if not incident_files:
        return None
        
    all_data = []
    for file_path in incident_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_data.append(json.load(f))
    
    # Aggregate statistics
    total_videos = len(all_data)
    total_motion_incidents = sum(d['summary']['motion_incidents'] for d in all_data)
    total_damage_incidents = sum(d['summary']['damage_incidents'] for d in all_data)
    total_frames = sum(d['summary']['total_frames'] for d in all_data)
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_videos_processed': total_videos,
        'total_frames_processed': total_frames,
        'total_motion_incidents': total_motion_incidents,
        'total_damage_incidents': total_damage_incidents,
        'video_summaries': [d['summary'] for d in all_data],
        'all_damage_events': [
            event 
            for d in all_data 
            for event in d.get('damage_incidents', [])
        ]
    }
    
    # Save consolidated report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        cfg.INCIDENT_LOG_DIR,
        f"consolidated_report_{timestamp}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger = get_logger()
    logger.info("Consolidated report saved: %s", output_file)
    
    return output_file
