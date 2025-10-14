from __future__ import annotations

import cv2
from .config import get_logger


class VideoCapture:
    def __init__(self, source: str | int):
        self.logger = get_logger()
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")

    def read(self):
        if not self.cap:
            self.logger.error("VideoCapture not initialized properly.")
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            self.logger.debug("End of video or read failed.")
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()

    def get_fps(self) -> float:
        try:
            return float(self.cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            return 30.0

    def get_frame_size(self) -> tuple[int, int]:
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        except Exception:
            return 0, 0
