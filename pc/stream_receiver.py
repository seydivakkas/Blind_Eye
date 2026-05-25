"""
pc/stream_receiver.py
═════════════════════
HTTP/MJPEG Video Stream Alıcı + Thread-Safe Ring Buffer

Raspberry Pi 3 B+'dan gelen MJPEG stream'i HTTP üzerinden alır,
frame'leri decode edip ring buffer'a yazar.

Mimari:
    Pi (MJPEG/HTTP) ──WiFi──▶ StreamReceiver.ring_buffer
                                      │
                                      ▼
                              preprocess.py → vsr_engine.py

Protokol:
    - HTTP GET /video → multipart/x-mixed-replace MJPEG stream
    - Her frame: --boundary\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n<JPEG>
    - Fallback: Raw TCP (4-byte header + JPEG) — eski uyumluluk

Kullanım:
    receiver = StreamReceiver(pi_url="http://192.168.1.50:8080/video")
    receiver.start()
    frame = receiver.get_latest()      # Son frame (non-blocking)
    frames = receiver.get_batch(30)    # Son 30 frame (VSR girişi)
    receiver.stop()
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamReceiver:
    """HTTP/MJPEG frame alıcı + ring buffer.

    Parameters
    ----------
    pi_url : str
        Pi'nin MJPEG stream URL'si. Örn: "http://192.168.1.50:8080/video"
    buffer_size : int
        Ring buffer kapasitesi (frame sayısı). Varsayılan 90 = 6 saniyelik tampon @15fps.
    timeout_s : float
        HTTP bağlantı timeout süresi.
    reconnect_interval_s : float
        Bağlantı koparsa yeniden deneme aralığı.
    """

    def __init__(
        self,
        pi_url: str = "http://192.168.1.50:8080/video",
        buffer_size: int = 90,
        timeout_s: float = 10.0,
        reconnect_interval_s: float = 3.0,
    ):
        self.pi_url = pi_url
        self.buffer_size = buffer_size
        self.timeout_s = timeout_s
        self.reconnect_interval_s = reconnect_interval_s

        # Thread-safe ring buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # State
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # İstatistikler
        self.frames_received: int = 0
        self.bytes_received: int = 0
        self.connection_errors: int = 0
        self.actual_fps: float = 0.0
        self._fps_counter: int = 0
        self._fps_time: float = time.time()

    # ──────────── PUBLIC API ────────────

    def start(self):
        """Alıcıyı başlat — arka plan thread'i açar."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="stream-receiver"
        )
        self._thread.start()
        logger.info(f"StreamReceiver başlatıldı → {self.pi_url}")

    def stop(self):
        """Alıcıyı durdur."""
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(
            f"StreamReceiver durduruldu — {self.frames_received} frame alındı"
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_latest(self) -> Optional[np.ndarray]:
        """Ring buffer'daki en son frame'i döndür (non-blocking).

        Returns
        -------
        np.ndarray | None
            [H, W, 3] uint8 BGR formatında. Buffer boşsa None.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1].copy()

    def get_batch(self, n: int) -> Optional[List[np.ndarray]]:
        """Ring buffer'dan son N frame'i döndür.

        Parameters
        ----------
        n : int
            İstenen frame sayısı (tipik: 30 = 1 VSR chunk).

        Returns
        -------
        list[np.ndarray] | None
            [H, W, 3] uint8 BGR frame listesi. Yeterli frame yoksa None.
        """
        with self._lock:
            if len(self._buffer) < n:
                return None
            # Son n frame (kronolojik sıra korunur)
            return [f.copy() for f in list(self._buffer)[-n:]]

    def get_buffer_fill(self) -> float:
        """Buffer doluluk oranı (0.0 – 1.0)."""
        with self._lock:
            return len(self._buffer) / self.buffer_size

    def clear(self):
        """Buffer'ı temizle."""
        with self._lock:
            self._buffer.clear()
        logger.info("StreamReceiver buffer temizlendi")

    # ──────────── INTERNAL ────────────

    def _receive_loop(self):
        """Ana alıcı döngüsü — MJPEG stream'den frame oku."""
        while self._running:
            if not self._connected:
                self._connect()
                if not self._connected:
                    time.sleep(self.reconnect_interval_s)
                    continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("Stream frame okunamadı — yeniden bağlanılıyor")
                    self._connected = False
                    self._release_cap()
                    continue

                # Ring buffer'a ekle
                with self._lock:
                    self._buffer.append(frame)

                self.frames_received += 1
                self.bytes_received += frame.nbytes
                self._update_fps()

            except Exception as e:
                logger.error(f"Frame okuma hatası: {e}")
                self._connected = False
                self._release_cap()
                self.connection_errors += 1

    def _connect(self):
        """MJPEG stream'e HTTP bağlantısı kur."""
        try:
            logger.info(f"Pi'ye bağlanılıyor: {self.pi_url}")
            self._cap = cv2.VideoCapture(self.pi_url, cv2.CAP_FFMPEG)

            # OpenCV MJPEG timeout ayarı
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.timeout_s * 1000))
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if self._cap.isOpened():
                self._connected = True
                logger.info(
                    f"✅ Pi MJPEG stream bağlantısı kuruldu: {self.pi_url}"
                )
            else:
                logger.debug("MJPEG stream açılamadı")
                self._connected = False
                self.connection_errors += 1
                self._release_cap()

        except Exception as e:
            logger.debug(f"Bağlantı hatası: {e}")
            self._connected = False
            self.connection_errors += 1

    def _release_cap(self):
        """VideoCapture'ı serbest bırak."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _update_fps(self):
        """Gerçek FPS hesapla."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now
