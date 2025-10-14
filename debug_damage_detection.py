#!/usr/bin/env python3
"""Debug tool to visualize damage detection metrics in real-time."""
from __future__ import annotations

import argparse
import cv2
import numpy as np
from intrusion_detection import Config, DamageDetector, load_rois_from_file


def visualize_damage_detection(video_path: str, roi_file: str = None):
    """Show real-time damage detection metrics with adjustable thresholds."""
    
    cfg = Config()
    
    # Load ROIs
    if roi_file:
        doors, windows = load_rois_from_file(video_path)
        cfg.DOOR_ROIS = doors
        cfg.WINDOW_ROIS = windows
    
    if not cfg.DOOR_ROIS and not cfg.WINDOW_ROIS:
        print("No ROIs defined. Run select_rois.py first.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return
    
    # Trackbars for threshold adjustment
    cv2.namedWindow('Damage Debug')
    cv2.createTrackbar('Diff Thresh', 'Damage Debug', int(cfg.DAMAGE_DIFF_THRESHOLD), 100, lambda x: None)
    cv2.createTrackbar('Edge Spike x10', 'Damage Debug', int(cfg.DAMAGE_EDGE_SPIKE_RATIO * 10), 50, lambda x: None)
    cv2.createTrackbar('Frag Count', 'Damage Debug', cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD, 100, lambda x: None)
    cv2.createTrackbar('Persistence', 'Damage Debug', cfg.DAMAGE_MIN_PERSISTENCE, 10, lambda x: None)
    
    detector = DamageDetector(cfg)
    frame_idx = 0
    paused = False
    
    print("\nControls:")
    print("  SPACE = pause/resume")
    print("  R = reset detector")
    print("  Q = quit")
    print("  Adjust trackbars to tune sensitivity\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
            
            # Update config from trackbars
            cfg.DAMAGE_DIFF_THRESHOLD = float(cv2.getTrackbarPos('Diff Thresh', 'Damage Debug'))
            cfg.DAMAGE_EDGE_SPIKE_RATIO = cv2.getTrackbarPos('Edge Spike x10', 'Damage Debug') / 10.0
            cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD = cv2.getTrackbarPos('Frag Count', 'Damage Debug')
            cfg.DAMAGE_MIN_PERSISTENCE = cv2.getTrackbarPos('Persistence', 'Damage Debug')
            
            # Process
            events = detector.process_frame(frame)
            
            # Annotate
            display = frame.copy()
            
            # Draw ROIs and metrics
            for idx, (x, y, w, h) in enumerate(cfg.DOOR_ROIS):
                state = detector.door_states[idx]
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Extract ROI for metrics
                roi = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                
                if state['baseline'] is not None:
                    diff = cv2.absdiff(gray, state['baseline'])
                    diff_mean = float(np.mean(diff))
                else:
                    diff_mean = 0.0
                
                edges = cv2.Canny(gray, 60, 150)
                edge_count = int(np.count_nonzero(edges))
                edge_avg = state['edge_avg']
                spike_ratio = edge_count / (edge_avg + 1e-6)
                
                # Display metrics
                y_offset = y - 70
                cv2.putText(display, f"DOOR {idx}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(display, f"Diff: {diff_mean:.1f} (thr:{cfg.DAMAGE_DIFF_THRESHOLD:.1f})", 
                           (x, y_offset+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display, f"Edge: {edge_count} avg:{edge_avg:.0f} spike:{spike_ratio:.2f} (thr:{cfg.DAMAGE_EDGE_SPIKE_RATIO:.1f})", 
                           (x, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display, f"Persist: {state['persist']}/{cfg.DAMAGE_MIN_PERSISTENCE}", 
                           (x, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Trigger indicator
                candidate = (diff_mean >= cfg.DAMAGE_DIFF_THRESHOLD and spike_ratio >= cfg.DAMAGE_EDGE_SPIKE_RATIO)
                color = (0, 255, 0) if candidate else (100, 100, 100)
                cv2.circle(display, (x + w + 10, y + 10), 8, color, -1)
            
            for idx, (x, y, w, h) in enumerate(cfg.WINDOW_ROIS):
                state = detector.window_states[idx]
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                edges = cv2.Canny(gray, 60, 150)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                small_cnts = [c for c in cnts if 5 < cv2.contourArea(c) < 150]
                fragment_count = len(small_cnts)
                edge_count = int(np.count_nonzero(edges))
                edge_avg = state['edge_avg']
                spike_ratio = edge_count / (edge_avg + 1e-6)
                
                y_offset = y - 70
                cv2.putText(display, f"WINDOW {idx}", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(display, f"Frags: {fragment_count} (thr:{cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD})", 
                           (x, y_offset+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display, f"Spike: {spike_ratio:.2f} (thr:{cfg.DAMAGE_EDGE_SPIKE_RATIO:.1f})", 
                           (x, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display, f"Persist: {state['persist']}/{cfg.DAMAGE_MIN_PERSISTENCE}", 
                           (x, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                candidate = (fragment_count >= cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD and spike_ratio >= cfg.DAMAGE_EDGE_SPIKE_RATIO)
                color = (0, 255, 0) if candidate else (100, 100, 100)
                cv2.circle(display, (x + w + 10, y + 10), 8, color, -1)
            
            # Show events
            if events:
                for ev in events:
                    print(f"Frame {frame_idx}: {ev.roi_type} {ev.event_type} score={ev.score:.1f}")
                    cv2.putText(display, f"EVENT: {ev.roi_type} {ev.event_type}", 
                               (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.putText(display, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Damage Debug', display)
            frame_idx += 1
        
        key = cv2.waitKey(30 if not paused else 1) & 0xFF
        
        if key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            detector = DamageDetector(cfg)
            print("Detector reset")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug damage detection")
    parser.add_argument("video", help="Video file path")
    args = parser.parse_args()
    
    visualize_damage_detection(args.video, roi_file=True)
