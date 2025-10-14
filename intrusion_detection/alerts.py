from __future__ import annotations

import time
from .config import get_logger, Config


class SimpleAlertSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger()
        self.last_alert_time = 0.0

    def notify(self, message: str):
        now = time.time()
        if now - self.last_alert_time >= self.cfg.ALERT_COOLDOWN:
            self.logger.info("ALERT: %s", message)
            self.last_alert_time = now
