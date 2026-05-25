"""
Blind Eye — WiFi Stream Server (PC Tarafı)
============================================
Pi 3 Model B+'dan gelen video stream'i alır,
pipeline'a frame verir, sonucu Pi'ye geri gönderir.

Mimari:
    Pi StreamClient ──TCP──▶ StreamServer ──▶ Pipeline
                                                │
    Pi SubtitleReceiver ◀──TCP──── Subtitle ◀───┘

Kullanım:
    server = StreamServer(port=8554, subtitle_port=8555)
    server.on_frame = lambda frame: pipeline.process(frame)
    server.start()
    # ...
    server.send_subtitle("merhaba", confidence=0.85)
    server.stop()
"""

import json
import logging
import socket
import struct
import threading
import time
import numpy as np
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StreamServer:
    """
    WiFi video stream alıcı + altyazı gönderici.

    Pi'den MJPEG frame alır, callback ile pipeline'a verir.
    Pipeline sonucunu (altyazı) Pi'ye TCP ile geri gönderir.

    Parameters
    ----------
    port : int
        Video stream dinleme portu.
    subtitle_port : int
        Altyazı gönderme portu.
    bind_ip : str
        Dinleme adresi.
    """

    def __init__(
        self,
        port: int = 8554,
        subtitle_port: int = 8555,
        bind_ip: str = "0.0.0.0",
    ):
        self.port = port
        self.subtitle_port = subtitle_port
        self.bind_ip = bind_ip

        # Sockets
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._subtitle_socket: Optional[socket.socket] = None

        # State
        self._running = False
        self._connected = False
        self._stream_thread = None
        self._subtitle_thread = None

        # Callback
        self.on_frame: Optional[Callable[[np.ndarray], None]] = None

        # İstatistikler
        self.frames_received = 0
        self.bytes_received = 0
        self.subtitles_sent = 0
        self.actual_fps = 0.0
        self._fps_counter = 0
        self._fps_time = time.time()

    def start(self):
        """Sunucuyu başlat."""
        if self._running:
            return

        self._running = True

        # Video stream thread
        self._stream_thread = threading.Thread(
            target=self._stream_listen_loop, daemon=True, name="stream-server"
        )
        self._stream_thread.start()

        # Subtitle sender thread
        self._subtitle_thread = threading.Thread(
            target=self._subtitle_connect_loop, daemon=True, name="subtitle-sender"
        )
        self._subtitle_thread.start()

        logger.info(
            f"StreamServer başlatıldı: video={self.port}, subtitle={self.subtitle_port}"
        )

    def stop(self):
        """Sunucuyu durdur."""
        self._running = False
        self._close_all()

        for t in [self._stream_thread, self._subtitle_thread]:
            if t:
                t.join(timeout=3.0)

        logger.info(
            f"StreamServer durduruldu — {self.frames_received} frame alındı, "
            f"{self.subtitles_sent} altyazı gönderildi"
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_subtitle(self, text: str, confidence: float = 0.0,
                       expression: str = ""):
        """
        Altyazı metnini Pi'ye gönder.

        Parameters
        ----------
        text : str
            Altyazı metni.
        confidence : float
            Güven skoru (0-1).
        expression : str
            Mimik durumu (opsiyonel).
        """
        if self._subtitle_socket is None:
            return

        msg = {
            "type": "subtitle",
            "text": text,
            "confidence": confidence,
            "expression": expression,
            "timestamp": time.time(),
        }

        try:
            data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
            header = struct.pack(">I", len(data))
            self._subtitle_socket.sendall(header + data)
            self.subtitles_sent += 1
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Altyazı gönderme hatası: {e}")
            self._subtitle_socket = None

    def send_expression(self, expression: str):
        """Mimik durumunu Pi'ye gönder."""
        if self._subtitle_socket is None:
            return

        msg = {
            "type": "expression",
            "expression": expression,
            "timestamp": time.time(),
        }

        try:
            data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
            header = struct.pack(">I", len(data))
            self._subtitle_socket.sendall(header + data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._subtitle_socket = None

    # ──────────── VIDEO STREAM ────────────

    def _stream_listen_loop(self):
        """Video stream dinleme döngüsü."""
        while self._running:
            try:
                self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )
                self._server_socket.settimeout(2.0)
                self._server_socket.bind((self.bind_ip, self.port))
                self._server_socket.listen(1)

                logger.info(f"Pi video bağlantısı bekleniyor: port {self.port}")

                while self._running:
                    try:
                        client, addr = self._server_socket.accept()
                        self._client_socket = client
                        self._connected = True
                        logger.info(f"Pi bağlandı (video): {addr}")
                        self._receive_frames()
                    except socket.timeout:
                        continue
                    except OSError:
                        if self._running:
                            break

            except Exception as e:
                logger.error(f"Stream server hatası: {e}")
                time.sleep(2.0)
            finally:
                self._connected = False
                if self._server_socket:
                    try:
                        self._server_socket.close()
                    except Exception:
                        pass

    def _receive_frames(self):
        """Bağlı Pi'den frame al."""
        import cv2

        while self._running and self._connected:
            try:
                # 4 byte boyut header
                header = self._recv_exact(self._client_socket, 4)
                if header is None:
                    break

                frame_size = struct.unpack(">I", header)[0]
                if frame_size > 5_000_000:  # 5MB güvenlik limiti
                    logger.warning(f"Çok büyük frame: {frame_size} bytes")
                    break

                # JPEG verisi
                jpeg_data = self._recv_exact(self._client_socket, frame_size)
                if jpeg_data is None:
                    break

                self.bytes_received += frame_size

                # JPEG → numpy array
                np_data = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.frames_received += 1
                    self._update_fps()

                    # Callback — pipeline'a ver
                    if self.on_frame:
                        self.on_frame(frame)

            except (ConnectionResetError, BrokenPipeError):
                logger.info("Pi video bağlantısı koptu")
                break
            except Exception as e:
                logger.error(f"Frame alma hatası: {e}")
                break

        self._connected = False
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass

    # ──────────── SUBTITLE SENDER ────────────

    def _subtitle_connect_loop(self):
        """Pi'nin subtitle receiver'ına bağlan."""
        while self._running:
            if self._subtitle_socket is None and self._connected:
                # Pi'nin IP'sini video bağlantısından al
                if self._client_socket:
                    try:
                        pi_ip = self._client_socket.getpeername()[0]
                        self._subtitle_socket = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM
                        )
                        self._subtitle_socket.settimeout(5.0)
                        self._subtitle_socket.connect(
                            (pi_ip, self.subtitle_port)
                        )
                        self._subtitle_socket.settimeout(None)
                        logger.info(
                            f"Pi subtitle bağlantısı kuruldu: {pi_ip}:{self.subtitle_port}"
                        )
                    except Exception as e:
                        logger.debug(f"Subtitle bağlantısı başarısız: {e}")
                        self._subtitle_socket = None

            time.sleep(2.0)

    # ──────────── UTILITY ────────────

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Tam n byte oku."""
        data = b""
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None
        return data

    def _update_fps(self):
        """FPS hesapla."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.actual_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now

    def _close_all(self):
        """Tüm soketleri kapat."""
        for s in [self._client_socket, self._server_socket, self._subtitle_socket]:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self._client_socket = None
        self._server_socket = None
        self._subtitle_socket = None
