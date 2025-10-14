from __future__ import annotations

import cv2
import numpy as np
from .config import Config, get_logger


class BackgroundSubtractor:
    def __init__(self, cfg: Config, method: str = 'KNN') -> None:
        self.logger = get_logger()
        self.cfg = cfg
        method = method.upper()
        if method == 'KNN':
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=cfg.BACKGROUND_HISTORY,
                dist2Threshold=cfg.BACKGROUND_THRESHOLD,
                detectShadows=True,
            )
        elif method == 'MOG2':
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=cfg.BACKGROUND_HISTORY,
                varThreshold=cfg.BACKGROUND_THRESHOLD,
                detectShadows=True,
            )
        else:
            self.logger.warning("Unknown method %s, defaulting to KNN", method)
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=cfg.BACKGROUND_HISTORY,
                dist2Threshold=cfg.BACKGROUND_THRESHOLD,
                detectShadows=True,
            )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.subtractor.apply(frame)
        # Remove shadows (shadow value is 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return fg_mask
