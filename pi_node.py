"""
pi_node.py
══════════
Raspberry Pi 3 B+ Ana Gözlük Node — Video Stream + OLED + Ağ Yönetimi

Roller:
    1. Pi Camera'dan MJPEG/HTTP video stream gönder (WiFi → PC)
    2. PC'den MQTT ile gelen altyazıyı OLED ekranda göster
    3. Ağ bağlantısını izle, otomatik yeniden bağlan
    4. SES YOK — tamamen görsel pipeline

Donanım:
    - Raspberry Pi 3 B+ (1.2GHz quad-core, 1GB RAM)
    - Pi Camera Module v2 (Sony IMX219)
    - SSD1306 OLED (128×64, I2C)

Kullanım:
    python pi_node.py
    python pi_node.py --pc-ip 192.168.1.100 --res 320x240 --fps 15

Protokol:
    Pi ──MJPEG/HTTP──▶ PC (StreamReceiver)
    Pi ◀──MQTT──────── PC (OledSender)
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger("pi_node")

# ── Platform kontrolü ──
IS_PI = os.path.exists("/proc/device-tree/model")

# ── Kamera import ──
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    _PICAM_AVAILABLE = True
except ImportError:
    _PICAM_AVAILABLE = False

# ── OLED import ──
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from PIL import ImageFont, Image, ImageDraw
    _OLED_AVAILABLE = True
except ImportError:
    _OLED_AVAILABLE = False

# ── MQTT import ──
try:
    from pi.mqtt_subtitle_rx import MQTTSubtitleReceiver
    _MQTT_RX_AVAILABLE = True
except ImportError:
    try:
        from mqtt_subtitle_rx import MQTTSubtitleReceiver
        _MQTT_RX_AVAILABLE = True
    except ImportError:
        _MQTT_RX_AVAILABLE = False

# ── OpenCV fallback (PC'de test için) ──
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ═══════════════════════════════════════════════════════
# MJPEG HTTP Server
# ═══════════════════════════════════════════════════════

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """MJPEG multipart HTTP stream handler."""

    # Class-level: frame paylaşımı
    _frame: Optional[bytes] = None
    _frame_lock = threading.Lock()
    _frame_event = threading.Event()

    def do_GET(self):
        if self.path == "/video" or self.path == "/":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()

            while True:
                MJPEGStreamHandler._frame_event.wait(timeout=2.0)
                with MJPEGStreamHandler._frame_lock:
                    frame_data = MJPEGStreamHandler._frame

                if frame_data is None:
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    break
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = json.dumps({"status": "ok", "device": "pi"})
            self.wfile.write(status.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """HTTP loglarını sustur."""
        pass

    @classmethod
    def update_frame(cls, jpeg_bytes: bytes):
        """Yeni JPEG frame'i güncelle."""
        with cls._frame_lock:
            cls._frame = jpeg_bytes
        cls._frame_event.set()
        cls._frame_event.clear()


# ═══════════════════════════════════════════════════════
# OLED Controller
# ═══════════════════════════════════════════════════════

class OLEDController:
    """SSD1306 OLED ekran yöneticisi."""

    def __init__(self, width=128, height=64, i2c_port=1, i2c_address=0x3C):
        self._device = None
        self._font = None
        self._available = False

        if _OLED_AVAILABLE:
            try:
                serial = i2c(port=i2c_port, address=i2c_address)
                self._device = ssd1306(serial, width=width, height=height)
                self._device.contrast(200)
                try:
                    self._font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
                    )
                except Exception:
                    self._font = ImageFont.load_default()
                self._available = True
                logger.info("✅ OLED ekran hazır")
            except Exception as e:
                logger.warning(f"OLED başlatılamadı: {e}")
        else:
            logger.info("OLED kütüphaneleri yok — konsol modu")

    def show(self, line1: str, line2: str = "", confidence: float = 0.0):
        """2 satır metin göster."""
        if self._available and self._device:
            try:
                img = Image.new("1", (128, 64), "black")
                draw = ImageDraw.Draw(img)
                draw.text((2, 2), line1[:21], font=self._font, fill="white")
                draw.text((2, 22), line2[:21], font=self._font, fill="white")

                # Güven çubuğu (altta)
                bar_width = int(confidence * 124)
                draw.rectangle([(2, 52), (2 + bar_width, 60)], fill="white")

                self._device.display(img)
            except Exception as e:
                logger.debug(f"OLED yazma hatası: {e}")
        else:
            # Konsol fallback
            print(f"📺 [{confidence:.0%}] {line1} | {line2}")

    def clear(self):
        if self._available and self._device:
            try:
                self._device.clear()
            except Exception:
                pass

    def show_status(self, msg: str):
        """Tek satır durum mesajı göster."""
        self.show(msg, "")


# ═══════════════════════════════════════════════════════
# Pi Node Ana Sınıfı
# ═══════════════════════════════════════════════════════

class PiNode:
    """Raspberry Pi 3 B+ ana gözlük node'u.

    Parameters
    ----------
    pc_ip : str
        PC'nin IP adresi (MQTT broker).
    stream_port : int
        MJPEG HTTP stream portu.
    resolution : tuple
        Kamera çözünürlüğü (genişlik, yükseklik).
    fps : int
        Hedef FPS.
    jpeg_quality : int
        JPEG sıkıştırma kalitesi (1-100).
    """

    def __init__(
        self,
        pc_ip: str = "192.168.1.100",
        stream_port: int = 8080,
        resolution: tuple = (320, 240),
        fps: int = 15,
        jpeg_quality: int = 40,
    ):
        self.pc_ip = pc_ip
        self.stream_port = stream_port
        self.resolution = resolution
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._running = False
        self._cam = None
        self._cam_thread: Optional[threading.Thread] = None
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        # Modüller
        self._oled = OLEDController()
        self._mqtt_rx: Optional[MQTTSubtitleReceiver] = None

    def start(self):
        """Node'u başlat."""
        self._running = True
        logger.info("═══ Blind Eye Pi Node ═══")

        # OLED açılış mesajı
        self._oled.show_status("Baslatiliyor...")

        # 1. HTTP MJPEG Server
        self._start_http_server()

        # 2. Kamera → MJPEG frame
        self._start_camera()

        # 3. MQTT altyazı alıcı
        self._start_mqtt()

        self._oled.show_status("Hazir!")
        logger.info(f"✅ Pi Node başlatıldı — stream: :{self.stream_port}/video")

    def stop(self):
        """Node'u durdur."""
        logger.info("Pi Node durduruluyor...")
        self._running = False

        if self._mqtt_rx:
            self._mqtt_rx.stop()

        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass

        if self._http_server:
            self._http_server.shutdown()

        self._oled.show_status("Kapandi")
        self._oled.clear()

        logger.info("✅ Pi Node güvenli şekilde kapatıldı")

    # ──────────── Kamera ────────────

    def _start_camera(self):
        """Pi Camera veya OpenCV webcam başlat."""
        self._cam_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="pi-camera"
        )
        self._cam_thread.start()

    def _camera_loop(self):
        """Frame yakala → JPEG encode → HTTP handler'a ver."""
        if _PICAM_AVAILABLE and IS_PI:
            self._camera_loop_picam()
        elif _CV2_AVAILABLE:
            self._camera_loop_cv2()
        else:
            logger.error("Ne picamera2 ne de OpenCV bulunamadı!")

    def _camera_loop_picam(self):
        """Pi Camera Module v2 ile MJPEG frame üretimi."""
        try:
            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
            )
            self._cam.configure(config)
            self._cam.start()
            logger.info(f"Pi Camera başlatıldı: {self.resolution} @{self.fps}fps")

            import io
            interval = 1.0 / self.fps
            while self._running:
                t0 = time.time()
                frame = self._cam.capture_array()
                # BGR→JPEG
                import cv2 as cv
                _, jpeg = cv.imencode(
                    ".jpg", frame,
                    [cv.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                MJPEGStreamHandler.update_frame(jpeg.tobytes())
                elapsed = time.time() - t0
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Pi Camera hatası: {e}")

    def _camera_loop_cv2(self):
        """OpenCV webcam fallback (PC'de test için)."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"OpenCV webcam başlatıldı: {self.resolution}")

        interval = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if ret:
                _, jpeg = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                MJPEGStreamHandler.update_frame(jpeg.tobytes())
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

        cap.release()

    # ──────────── HTTP Server ────────────

    def _start_http_server(self):
        """MJPEG HTTP server başlat."""
        self._http_server = HTTPServer(
            ("0.0.0.0", self.stream_port), MJPEGStreamHandler
        )
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="mjpeg-http",
        )
        self._http_thread.start()
        logger.info(f"MJPEG HTTP Server: 0.0.0.0:{self.stream_port}")

    # ──────────── MQTT ────────────

    def _start_mqtt(self):
        """MQTT altyazı alıcıyı başlat."""
        if not _MQTT_RX_AVAILABLE:
            logger.warning("MQTT alıcı modülü bulunamadı")
            return

        self._mqtt_rx = MQTTSubtitleReceiver(
            broker_host=self.pc_ip,
            broker_port=1883,
        )
        self._mqtt_rx.on_subtitle = self._on_subtitle
        self._mqtt_rx.start()

    def _on_subtitle(self, line1: str, line2: str, confidence: float):
        """MQTT'den gelen altyazıyı OLED'e yaz."""
        self._oled.show(line1, line2, confidence)


# ═══════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Blind Eye — Raspberry Pi 3 B+ Gözlük Node (SES YOK)"
    )
    parser.add_argument(
        "--pc-ip", type=str, default="192.168.1.100",
        help="PC IP adresi (MQTT broker)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="MJPEG HTTP stream portu",
    )
    parser.add_argument(
        "--res", type=str, default="320x240",
        help="Kamera çözünürlüğü (WxH)",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Hedef FPS",
    )
    parser.add_argument(
        "--quality", type=int, default=40,
        help="JPEG kalitesi (1-100)",
    )
    args = parser.parse_args()

    # Çözünürlük parse
    try:
        w, h = args.res.split("x")
        resolution = (int(w), int(h))
    except ValueError:
        resolution = (320, 240)

    node = PiNode(
        pc_ip=args.pc_ip,
        stream_port=args.port,
        resolution=resolution,
        fps=args.fps,
        jpeg_quality=args.quality,
    )

    # Ctrl+C handler
    def signal_handler(sig, frame):
        logger.info("\nKapatılıyor...")
        node.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    node.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        node.stop()


if __name__ == "__main__":
    main()
