from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import Config, get_logger


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    conf: float

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class MotionDetector:
    def __init__(self, cfg: Config, person_detector=None):
        self.cfg = cfg
        self.logger = get_logger()
        self.person_detector = person_detector
        self.persistence_counter = 0
        self._last_bbox = None
        self._smooth_alpha = 0.25  # bbox smoothing factor (0..1)

    def _preprocess_for_person(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            # grayscale -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self.cfg.ENHANCE_CONTRAST:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_CLIP_LIMIT, tileGridSize=(self.cfg.CLAHE_TILE_SIZE, self.cfg.CLAHE_TILE_SIZE))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        if self.cfg.APPLY_GAMMA_CORRECTION and self.cfg.GAMMA != 1.0:
            gamma_lut = (np.array([((i / 255.0) ** (1.0 / self.cfg.GAMMA)) * 255 for i in range(256)]).astype("uint8"))
            frame = cv2.LUT(frame, gamma_lut)
        return frame

    def detect_persons_in_frame(self, frame: np.ndarray) -> List[Box]:
        if self.person_detector is None or not self.cfg.ENABLE_PERSON_DETECTION:
            return []
        try:
            processed = self._preprocess_for_person(frame)
            results = self.person_detector(processed, verbose=False)
            boxes: List[Box] = []
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                cls = getattr(r.boxes, 'cls', None)
                conf = getattr(r.boxes, 'conf', None)
                xyxy = getattr(r.boxes, 'xyxy', None)
                if xyxy is None or conf is None or cls is None:
                    continue
                for i in range(len(xyxy)):
                    try:
                        # person class id is 0 for COCO
                        if int(cls[i]) != 0:
                            continue
                        c = float(conf[i])
                        if c < self.cfg.PERSON_CONFIDENCE_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                        w, h = max(1, x2 - x1), max(1, y2 - y1)
                        area = w * h
                        if not (self.cfg.MIN_PERSON_AREA <= area <= self.cfg.MAX_PERSON_AREA):
                            continue
                        ar = w / float(h)
                        if not (self.cfg.PERSON_ASPECT_RATIO_MIN <= ar <= self.cfg.PERSON_ASPECT_RATIO_MAX):
                            continue
                        boxes.append(Box(x1, y1, w, h, c))
                    except Exception:
                        continue
            return boxes
        except Exception as e:
            self.logger.debug("YOLO detection failed: %s", e)
            return []

    @staticmethod
    def _overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        a_area = aw * ah
        b_area = bw * bh
        union = a_area + b_area - inter_area
        if union == 0:
            return 0.0
        return inter_area / float(union)

    def filter_motion_by_persons(self, contours: List[np.ndarray], person_boxes: List[Box]) -> List[np.ndarray]:
        if not person_boxes:
            return []
        filtered = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            for pb in person_boxes:
                if self._overlap_ratio((x, y, w, h), pb.to_tuple()) >= self.cfg.MIN_OVERLAP_RATIO:
                    filtered.append(c)
                    break
        return filtered

    def detect(self, frame: np.ndarray, fg_mask: np.ndarray):
        h, w = frame.shape[:2]
        min_contour_area = int(self.cfg.MIN_CONTOUR_AREA_PERCENT * w * h)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Stronger morphology to reduce noise
        fg_refined = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_refined = cv2.morphologyEx(fg_refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        fg_refined = cv2.dilate(fg_refined, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

        person_boxes = self.detect_persons_in_frame(frame)

        if self.cfg.ENABLE_PERSON_DETECTION:
            if person_boxes:
                filtered = self.filter_motion_by_persons(contours, person_boxes)
                if filtered:
                    contours = filtered
                else:
                    # Fallback: if motion overlap is too low, create synthetic contours from person boxes
                    contours = [np.array([[[pb.x, pb.y]], [[pb.x+pb.w, pb.y]], [[pb.x+pb.w, pb.y+pb.h]], [[pb.x, pb.y+pb.h]]], dtype=np.int32) for pb in person_boxes]
            if self.cfg.PERSON_ONLY_MODE and not person_boxes:
                contours = []  # enforce person-only tracking

        motion_detected = len(contours) > 0
        if motion_detected:
            self.persistence_counter = min(self.cfg.DETECTION_PERSISTENCE, self.persistence_counter + 1)
        else:
            self.persistence_counter = max(0, self.persistence_counter - 1)
            motion_detected = self.persistence_counter > 0

        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            # Smooth bbox to reduce jitter
            if self._last_bbox is not None:
                lx, ly, lw, lh = self._last_bbox
                x = int(lx * (1 - self._smooth_alpha) + x * self._smooth_alpha)
                y = int(ly * (1 - self._smooth_alpha) + y * self._smooth_alpha)
                w = int(lw * (1 - self._smooth_alpha) + w * self._smooth_alpha)
                h = int(lh * (1 - self._smooth_alpha) + h * self._smooth_alpha)
            bbox = (x, y, w, h)
            self._last_bbox = bbox
        else:
            bbox = None
            self._last_bbox = None

        return motion_detected, contours, bbox, person_boxes

    @staticmethod
    def draw_motion(frame: np.ndarray, contours: List[np.ndarray], bbox, person_boxes: List[Box]):
        out = frame.copy()
        # Draw motion contours
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
        # Draw per-contour bbox to better handle multiple persons
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(out, 'INTRUDER', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Draw person boxes
        for pb in person_boxes:
            x, y, w, h = pb.to_tuple()
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, f"person {pb.conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out
