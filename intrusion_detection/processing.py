from __future__ import annotations

import os
from typing import List, Optional

import cv2
import imageio
import numpy as np

from .config import Config, get_logger
from .video import VideoCapture
from .background import BackgroundSubtractor
from .motion import MotionDetector
from .damage import DamageDetector
from .roi_selector import load_rois_from_file
from .incidents import IncidentRecorder, generate_summary_report
from .alerts import SimpleAlertSystem
import time


def create_full_enhanced_video_with_imageio(
    input_video_path: str,
    cfg: Config,
    person_detector=None,
    max_frames: Optional[int] = None,
    alert_system: Optional[SimpleAlertSystem] = None,
) -> Optional[str]:
    logger = get_logger()
    cap = VideoCapture(input_video_path)
    if not cap.cap or not cap.cap.isOpened():
        logger.error("Failed to open input video: %s", input_video_path)
        return None

    # Auto-load ROIs from file if damage detection enabled
    if cfg.ENABLE_DAMAGE_DETECTION:
        doors, windows = load_rois_from_file(input_video_path)
        if doors or windows:
            cfg.DOOR_ROIS = doors
            cfg.WINDOW_ROIS = windows
            logger.info("Loaded %d door(s) and %d window(s) ROIs for this video", len(doors), len(windows))

    # Initialize incident recorder
    incident_recorder = IncidentRecorder(cfg, input_video_path) if cfg.ENABLE_INCIDENT_LOGGING else None
    
    # Use provided alert system or create new one
    if alert_system is None:
        alert_system = SimpleAlertSystem(cfg)

    fps = cap.get_fps() or 30.0
    subtractor = BackgroundSubtractor(cfg, method='KNN')
    detector = MotionDetector(cfg, person_detector=person_detector)
    damage_detector = DamageDetector(cfg) if cfg.ENABLE_DAMAGE_DETECTION else None

    output_path = Config.get_output_video_path(input_video_path, suffix='enhanced')
    writer = imageio.get_writer(output_path, fps=fps)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
            # Denoise pre-pass to reduce background noise
            if cfg.DENOISE_BEFORE_BG and cfg.DENOISE_KERNEL_SIZE % 2 == 1:
                frame = cv2.GaussianBlur(frame, (cfg.DENOISE_KERNEL_SIZE, cfg.DENOISE_KERNEL_SIZE), cfg.DENOISE_SIGMA)
            fg_mask = subtractor.apply(frame)
            motion_detected, contours, bbox, person_boxes = detector.detect(frame, fg_mask)
            
            # Log motion/intrusion events
            if motion_detected:
                event_type = 'person_detected' if person_boxes else 'motion_detected'
                person_count = len(person_boxes) if person_boxes else 0
                # Handle Box objects properly
                bboxes = []
                if person_boxes:
                    for box in person_boxes:
                        if hasattr(box, 'to_tuple'):
                            bboxes.append(list(box.to_tuple()))
                        else:
                            # Fallback for tuple format
                            bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                
                # Log to incident recorder
                if incident_recorder:
                    incident_recorder.log_motion_event(
                        frame_number=frame_count,
                        event_type=event_type,
                        person_count=person_count,
                        bounding_boxes=bboxes
                    )
                
                # Send Telegram alert if enabled
                if cfg.TELEGRAM_NOTIFY_ON_MOTION:
                    alert_system.notify_motion_detected(
                        frame_number=frame_count,
                        person_count=person_count,
                        bounding_boxes=bboxes,
                        frame_image=drawn if cfg.TELEGRAM_SEND_IMAGES else None
                    )
            
            drawn = MotionDetector.draw_motion(frame, contours, bbox, person_boxes)
            
            if damage_detector is not None:
                events = damage_detector.process_frame(frame)
                # Log damage events and send alerts
                if events:
                    for event in events:
                        # Log to incident recorder
                        if incident_recorder:
                            incident_recorder.log_damage_event(frame_count, event)
                        
                        # Send Telegram alert if enabled
                        if cfg.TELEGRAM_NOTIFY_ON_DAMAGE:
                            alert_system.notify_damage_detected(
                                frame_number=frame_count,
                                roi_type=event.roi_type,
                                roi_index=event.roi_index,
                                event_type=event.event_type,
                                score=event.score,
                                frame_image=drawn if cfg.TELEGRAM_SEND_IMAGES else None
                            )
                # Always annotate with recent events for persistent highlight
                drawn = damage_detector.annotate(drawn, events, cfg, getattr(damage_detector, 'recent_events', None))
            
            # Optional gamma correction to lift dark shading in grayscale videos
            if cfg.APPLY_GAMMA_TO_OUTPUT and cfg.GAMMA != 1.0:
                gamma_lut = (np.array([((i / 255.0) ** (1.0 / cfg.GAMMA)) * 255 for i in range(256)]).astype("uint8"))
                drawn = cv2.LUT(drawn, gamma_lut)
            writer.append_data(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        writer.close()

    # Finalize incident report
    incident_log_path = None
    motion_count = 0
    damage_count = 0
    
    if incident_recorder:
        incident_log_path = incident_recorder.finalize(output_path, frame_count)
        motion_count = len(incident_recorder.motion_incidents)
        damage_count = len(incident_recorder.damage_incidents)

    processing_duration = time.time() - start_time
    
    logger.info("Saved enhanced video: %s (%d frames)", output_path, frame_count)
    if incident_log_path:
        logger.info("Incident log: %s", incident_log_path)
    
    # Send session summary via Telegram if enabled
    if cfg.TELEGRAM_NOTIFY_ON_SESSION_END:
        alert_system.notify_session_summary(
            video_file=os.path.basename(input_video_path),
            total_frames=frame_count,
            motion_incidents=motion_count,
            damage_incidents=damage_count,
            processing_duration=processing_duration
        )
    
    return output_path


def process_all_input_videos(
    cfg: Config,
    person_detector=None,
    selection: Optional[List[str]] = None,
    max_frames_per_video: Optional[int] = None,
    alert_system: Optional[SimpleAlertSystem] = None,
) -> List[str]:
    logger = get_logger()
    videos = []
    for name in os.listdir(cfg.INPUT_VIDEO_DIR):
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            videos.append(os.path.join(cfg.INPUT_VIDEO_DIR, name))
    if selection:
        # Filter only selected filenames (case-insensitive match on basename)
        sel_set = {os.path.basename(p).lower() for p in selection}
        videos = [v for v in videos if os.path.basename(v).lower() in sel_set]

    # Create shared alert system for all videos
    if alert_system is None:
        alert_system = SimpleAlertSystem(cfg)

    outputs: List[str] = []
    incident_logs: List[str] = []
    
    for v in sorted(videos):
        out = create_full_enhanced_video_with_imageio(
            v, cfg, 
            person_detector=person_detector, 
            max_frames=max_frames_per_video,
            alert_system=alert_system
        )
        if out:
            outputs.append(out)
            # Track incident log files
            if cfg.ENABLE_INCIDENT_LOGGING:
                base_name = os.path.splitext(os.path.basename(v))[0]
                # Find most recent incident log for this video
                incident_files = [
                    os.path.join(cfg.INCIDENT_LOG_DIR, f)
                    for f in os.listdir(cfg.INCIDENT_LOG_DIR)
                    if f.startswith(base_name) and f.endswith('.json')
                ]
                if incident_files:
                    incident_logs.append(max(incident_files, key=os.path.getmtime))
    
    logger.info("Processed %d/%d videos", len(outputs), len(videos))
    
    # Generate consolidated summary report if multiple videos processed
    if cfg.ENABLE_INCIDENT_LOGGING and len(incident_logs) > 1:
        summary_path = generate_summary_report(cfg, incident_logs)
        if summary_path:
            logger.info("Consolidated summary: %s", summary_path)
    
    # Shutdown alert system (sends shutdown notification if enabled)
    if alert_system:
        alert_system.shutdown()
    
    return outputs
