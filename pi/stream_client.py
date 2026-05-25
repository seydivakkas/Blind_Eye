"""
Blind Eye — WiFi Stream Client (Pi Tarafı)
===========================================
Pi 3 Model B+ üzerinde kameradan TCP socket stream gönderici.

Protokol: Saf TCP socket (HTTP/MJPEG DEĞİL)
    - Her frame: [4 byte boyut header (big-endian)] + [JPEG verisi]
    - Düşük latency, minimal overhead
    - Auto-reconnect: PC bağlantısı koparsa otomatik yeniden bağlanır

Mimari:
    PiCamera → JPEG encode → TCP socket → PC StreamServer

Kullanım:
    client = StreamClient(server_ip="192.168.1.100", port=8554)
    client.start(camera)  # PiCamera nesnesi
    # ... çalışırken ...
    client.stop()
"""

import logging
import threading
import time
import struct
import socket
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class StreamClient:
    """
    WiFi TCP socket stream gönderici.

    Pi kamerasından frame alır, JPEG'e sıkıştırır,
    saf TCP socket üzerinden PC'ye gönderir.

    Protokol (her frame):
        [4 bytes: frame boyutu (big-endian uint32)]
        [N bytes: JPEG verisi]

    Parameters
    ----------
    server_ip : str
        PC'nin IP adresi.
    port : int
        Stream portu.
    jpeg_quality : int
        JPEG sıkıştırma kalitesi (1-100).
    reconnect_interval : float
        Bağlantı koparsa yeniden deneme süresi (saniye).
    """

    def __init__(
        self,
        server_ip: str = "192.168.1.100",
        port: int = 8554,
        jpeg_quality: int = 70,
        reconnect_interval: float = 3.0,
    ):
        self.server_ip = server_ip
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.reconnect_interval = reconnect_interval

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread = None
        self._camera = None
        self._connected = False

        # İstatistikler
        self.frames_sent = 0
        self.bytes_sent = 0
        self.connection_errors = 0
        self.last_latency_ms = 0.0

    def start(self, camera):
        """
        Stream gönderimini başlat.

        Parameters
        ----------
        camera : PiCamera
            Kamera nesnesi (pi_camera.py).
        """
        if self._running:
            return

        self._camera = camera
        self._running = True
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="stream-client"
        )
        self._thread.start()
        logger.info(f"StreamClient başlatıldı → {self.server_ip}:{self.port}")

    def stop(self):
        """Stream gönderimini durdur."""
        self._running = False
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(f"StreamClient durduruldu — {self.frames_sent} frame gönderildi")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _stream_loop(self):
        """Ana stream döngüsü — bağlan + gönder."""
        while self._running:
            if not self._connected:
                self._connect()
                if not self._connected:
                    time.sleep(self.reconnect_interval)
                    continue

            # Frame oku ve gönder
            frame = self._camera.read() if self._camera else None
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                t0 = time.time()
                self._send_frame(frame)
                self.last_latency_ms = (time.time() - t0) * 1000
                self.frames_sent += 1
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logger.warning(f"Stream bağlantısı koptu: {e}")
                self._connected = False
                self._disconnect()
                self.connection_errors += 1

            # FPS kontrolü (kamera zaten kontrol ediyor ama ek güvenlik)
            time.sleep(0.001)

    def _connect(self):
        """PC'ye TCP bağlantısı kur."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.server_ip, self.port))
            self._socket.settimeout(None)
            self._connected = True
            logger.info(f"PC'ye bağlandı: {self.server_ip}:{self.port}")
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.debug(f"Bağlantı başarısız: {e}")
            self._connected = False
            self.connection_errors += 1

    def _disconnect(self):
        """Bağlantıyı kapat."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False

    def _send_frame(self, frame: np.ndarray):
        """
        Frame'i JPEG olarak sıkıştırıp TCP ile gönder.

        Protokol: [4 bytes boyut][JPEG data]
        """
        import cv2

        # JPEG encode
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, jpeg_data = cv2.imencode(".jpg", frame, encode_param)
        if not success:
            return

        data = jpeg_data.tobytes()
        size = len(data)
        self.bytes_sent += size

        # Boyut header + veri gönder
        header = struct.pack(">I", size)
        self._socket.sendall(header + data)
