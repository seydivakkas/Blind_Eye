"""
Blind Eye — Pi Camera Module
=============================
Raspberry Pi 3 Model B+ üzerinde kamera kontrolü.

Donanım: Pi Camera Module v2 (Sony IMX219, 8MP, 1080p30)
    Pi 3 Model B+ standart 15-pin CSI portu kullanır.
    Pi Camera Module v2 kutudaki kablo ile doğrudan bağlanır.
    Ek dönüştürücü kablo GEREKMEZ.

Desteklenen kamera modülleri:
    1. Pi Camera Module v2 (Sony IMX219, 8MP, 1080p30) ← KULLANILAN
    2. USB webcam (OpenCV fallback — PC test için)

Kullanım:
    cam = PiCamera(source=0, resolution=(640, 480), fps=25)
    cam.start()
    frame = cam.read()   # np.ndarray [480, 640, 3] uint8
    cam.stop()
"""

import logging
import threading
import time
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PiCamera:
    """
    Pi kamera yöneticisi.

    picamera2 varsa Pi Camera Module kullanır,
    yoksa OpenCV VideoCapture ile USB cam'e fallback yapar.

    Parameters
    ----------
    source : int | str
        Kamera kaynağı. 0 = varsayılan kamera.
    resolution : tuple
        (genişlik, yükseklik). Önerilen: (640, 480) veya (320, 240).
    fps : int
        Hedef FPS.
    rotation : int
        Döndürme: 0, 90, 180, 270.
    hflip : bool
        Yatay aynalama.
    vflip : bool
        Dikey aynalama.
    """

    def __init__(
        self,
        source: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 20,
        rotation: int = 0,
        hflip: bool = False,
        vflip: bool = False,
    ):
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.rotation = rotation
        self.hflip = hflip
        self.vflip = vflip

        self._cap = None
        self._picam2 = None
        self._backend = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

        # İstatistikler
        self.frame_count = 0
        self.actual_fps = 0.0
        self._fps_time = time.time()
        self._fps_counter = 0

    def start(self):
        """Kamerayı başlat."""
        if self._running:
            return

        self._backend = self._detect_backend()
        logger.info(f"PiCamera başlatılıyor: backend={self._backend}, "
                     f"res={self.resolution}, fps={self.fps}")

        if self._backend == "picamera2":
            self._start_picamera2()
        elif self._backend == "opencv":
            self._start_opencv()
        else:
            logger.warning("Kamera bulunamadı — MOCK mod aktif")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Kamerayı durdur."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

        if self._backend == "picamera2" and self._picam2:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
        elif self._backend == "opencv" and self._cap:
            self._cap.release()

        logger.info(f"PiCamera durduruldu — toplam {self.frame_count} frame")

    def read(self) -> Optional[np.ndarray]:
        """
        Son yakalanan frame'i oku.

        Returns
        -------
        np.ndarray | None
            [H, W, 3] uint8 BGR formatında. Frame yoksa None.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self) -> bool:
        return self._running

    # ──────────── BACKEND ────────────

    def _detect_backend(self) -> str:
        """Mevcut kamera kütüphanesini algıla."""
        try:
            from picamera2 import Picamera2  # noqa: F401
            return "picamera2"
        except ImportError:
            pass

        try:
            import cv2
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.release()
                return "opencv"
        except Exception:
            pass

        return "mock"

    def _start_picamera2(self):
        """picamera2 ile Pi Camera başlat."""
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"},
        )
        self._picam2.configure(config)
        self._picam2.start()
        logger.info("picamera2 başlatıldı")

    def _start_opencv(self):
        """OpenCV ile USB/dahili kamera başlat."""
        import cv2

        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"OpenCV kamera başlatıldı: source={self.source}")

    def _capture_loop(self):
        """Kamera yakalama döngüsü."""
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            frame = self._grab_frame()

            if frame is not None:
                # Döndürme
                frame = self._apply_transforms(frame)

                with self._lock:
                    self._frame = frame
                self.frame_count += 1
                self._update_fps()

            # FPS sınırlama
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Tek frame yakala."""
        if self._backend == "picamera2":
            try:
                frame = self._picam2.capture_array()
                # picamera2 RGB döner, BGR'ye çevir (OpenCV uyumlu)
                import cv2
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.debug(f"picamera2 frame hatası: {e}")
                return None

        elif self._backend == "opencv":
            ret, frame = self._cap.read()
            return frame if ret else None

        else:
            # Mock: gradient test frame
            h, w = self.resolution[1], self.resolution[0]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Sinüsoidal gradient
            t = time.time()
            x = np.linspace(0, 2 * np.pi, w)
            y = np.linspace(0, 2 * np.pi, h)
            X, Y = np.meshgrid(x, y)
            frame[:, :, 0] = ((np.sin(X + t) + 1) * 60).astype(np.uint8)
            frame[:, :, 1] = ((np.sin(Y + t * 0.5) + 1) * 80).astype(np.uint8)
            frame[:, :, 2] = ((np.sin(X + Y + t * 0.3) + 1) * 50).astype(np.uint8)
            return frame

    def _apply_transforms(self, frame: np.ndarray) -> np.ndarray:
        """Döndürme ve aynalama uygula."""
        import cv2

        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.hflip:
            frame = cv2.flip(frame, 1)
        if self.vflip:
            frame = cv2.flip(frame, 0)

        return frame

    def _update_fps(self):
        """FPS hesapla."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now
