import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CameraManager:
    """Kamera yakalama, FPS sabitleme, hata toleransı."""

    def __init__(self, device_id: int = 0, fps: int = 30,
                 resolution: Tuple[int, int] = (640, 480)):
        self.device_id = device_id
        self.fps = fps
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error(f"Kamera açılamadı: {self.device_id}")
            return False
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.running = True
        logger.info(
            f"Kamera başlatıldı: {self.resolution[0]}x{self.resolution[1]} "
            f"@ {self.fps}FPS"
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.running:
            return False, None
        return self.cap.read()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Kamera kapatıldı.")
