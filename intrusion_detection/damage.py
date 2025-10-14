from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np

from .config import Config, get_logger


@dataclass
class DamageEvent:
    roi_type: str  # 'door' or 'window'
    roi_index: int
    event_type: str  # 'forced_entry' | 'shatter'
    frame_index: int
    score: float


class DamageDetector:
    """Heuristic-based structural change detector.

    Strategy:
    - Maintain background snapshot for each ROI (door/window).
    - For each frame compute:
        * Mean absolute difference vs baseline.
        * Edge map (Canny) and edge pixel count.
        * Fragmentation: count of small contours (possible shattered pieces or splinters).
    - Trigger conditions:
        * Forced Entry (door): sustained high mean diff + edge spike.
        * Shatter (window): spike in small fragment count + edge spike.
    - Persistence and cooldown avoid duplicate alerts.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger()
        self.door_states = self._init_states(cfg.DOOR_ROIS, 'door')
        self.window_states = self._init_states(cfg.WINDOW_ROIS, 'window')
        self.frame_index = 0
        # Keep track of recent events for persistent highlighting
        self.recent_events: List[DamageEvent] = []

    def _init_states(self, rois: List[Tuple[int, int, int, int]], roi_type: str):
        states = []
        for _ in rois:
            states.append({
                'baseline': None,
                'edge_avg': 0.0,
                'edge_alpha': 0.1,
                'persist': 0,
                'last_event_frame': -10_000,
            })
        if rois:
            self.logger.info("Initialized %d %s ROI(s) for damage detection", len(rois), roi_type)
        return states

    def _extract_roi(self, frame, roi):
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]

    def _compute_features(self, roi_img, state):
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img
        if state['baseline'] is None:
            state['baseline'] = gray.copy()
            diff_mean = 0.0
        else:
            diff = cv2.absdiff(gray, state['baseline'])
            diff_mean = float(np.mean(diff))

        edges = cv2.Canny(gray, 60, 150)
        edge_count = int(np.count_nonzero(edges))
        # exponential moving average for baseline edge activity
        state['edge_avg'] = state['edge_alpha'] * edge_count + (1 - state['edge_alpha']) * state['edge_avg']

        # Fragmentation: small contours (potential shards)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_cnts = [c for c in cnts if 5 < cv2.contourArea(c) < 150]
        fragment_count = len(small_cnts)

        return diff_mean, edge_count, fragment_count

    def _evaluate_roi(self, roi_img, roi_type: str, idx: int, state: Dict) -> List[DamageEvent]:
        events: List[DamageEvent] = []
        diff_mean, edge_count, fragment_count = self._compute_features(roi_img, state)
        edge_spike = (state['edge_avg'] > 0 and edge_count / (state['edge_avg'] + 1e-6) >= self.cfg.DAMAGE_EDGE_SPIKE_RATIO)

        # Determine candidate event
        forced_entry_candidate = (roi_type == 'door' and diff_mean >= self.cfg.DAMAGE_DIFF_THRESHOLD and edge_spike)
        shatter_candidate = (roi_type == 'window' and fragment_count >= self.cfg.DAMAGE_FRAGMENT_COUNT_THRESHOLD and edge_spike)

        candidate = forced_entry_candidate or shatter_candidate
        if candidate:
            state['persist'] += 1
        else:
            state['persist'] = max(0, state['persist'] - 1)

        confirmed = state['persist'] >= self.cfg.DAMAGE_MIN_PERSISTENCE
        cooldown_ok = (self.frame_index - state['last_event_frame']) >= self.cfg.DAMAGE_EVENT_COOLDOWN

        if confirmed and cooldown_ok:
            event_type = 'forced_entry' if forced_entry_candidate else 'shatter'
            score = diff_mean if forced_entry_candidate else fragment_count
            events.append(DamageEvent(roi_type=roi_type, roi_index=idx, event_type=event_type, frame_index=self.frame_index, score=score))
            state['last_event_frame'] = self.frame_index
            state['persist'] = 0
            # Update baseline after event to adapt
            state['baseline'] = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if roi_img.ndim == 3 else roi_img.copy()
        return events

    def process_frame(self, frame) -> List[DamageEvent]:
        if not self.cfg.ENABLE_DAMAGE_DETECTION:
            return []
        events: List[DamageEvent] = []
        # Doors
        for idx, roi in enumerate(self.cfg.DOOR_ROIS):
            roi_img = self._extract_roi(frame, roi)
            events.extend(self._evaluate_roi(roi_img, 'door', idx, self.door_states[idx]))
        # Windows
        for idx, roi in enumerate(self.cfg.WINDOW_ROIS):
            roi_img = self._extract_roi(frame, roi)
            events.extend(self._evaluate_roi(roi_img, 'window', idx, self.window_states[idx]))
        # Append new events to recent list with timestamp
        if events:
            self.recent_events.extend(events)
        # Prune old events outside hold window
        hold = self.cfg.DAMAGE_EVENT_HOLD_FRAMES
        if hold > 0 and self.recent_events:
            self.recent_events = [e for e in self.recent_events if (self.frame_index - e.frame_index) <= hold]
        self.frame_index += 1
        return events

    @staticmethod
    def annotate(frame, events: List[DamageEvent], cfg: Config, recent_events: List[DamageEvent] | None = None):
        out = frame.copy()
        color_map = {'door': (0, 0, 255), 'window': (255, 0, 0)}
        # Combine events with recent ones (avoid duplicate drawing if same frame)
        to_draw = list(events)
        if recent_events:
            # Add only those not already in current events
            existing_ids = {(e.roi_type, e.roi_index, e.frame_index) for e in events}
            for e in recent_events:
                key = (e.roi_type, e.roi_index, e.frame_index)
                if key not in existing_ids:
                    to_draw.append(e)
        for ev in to_draw:
            rois = cfg.DOOR_ROIS if ev.roi_type == 'door' else cfg.WINDOW_ROIS
            if ev.roi_index >= len(rois):
                continue
            x, y, w, h = rois[ev.roi_index]
            base_color = color_map.get(ev.roi_type, (0, 255, 255))
            # Fade color based on age (recent events lighter)
            age = 0
            if recent_events is not None:
                age = max(0, cfg.DAMAGE_EVENT_HOLD_FRAMES - (cfg.DAMAGE_EVENT_HOLD_FRAMES if ev.frame_index is None else 0))
            color = base_color
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{ev.roi_type}:{ev.event_type}" \
                f" score={ev.score:.1f}" if ev.event_type == 'forced_entry' else f" {ev.roi_type}:{ev.event_type} cnt={int(ev.score)}"
            cv2.putText(out, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out
