"""
Blind Eye — Subtitle Receiver (Pi Tarafı)
==========================================
PC'den gelen altyazı metnini WiFi üzerinden alır
ve OLED display'e yönlendirir. (Pi 3 Model B+)

Protokol (her mesaj):
    [4 bytes: mesaj boyutu (big-endian uint32)]
    [N bytes: UTF-8 encoded JSON]
    
JSON formatı:
    {
        "type": "subtitle",         # "subtitle" | "expression" | "status"
        "text": "merhaba",          # Altyazı metni
        "confidence": 0.85,         # Güven skoru
        "expression": "mutlu",      # Mimik durumu (opsiyonel)
        "timestamp": 1716571234.5   # Unix timestamp
    }

Kullanım:
    receiver = SubtitleReceiver(port=8555)
    receiver.on_subtitle = lambda text, conf: oled.show(text)
    receiver.start()
"""

import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SubtitleReceiver:
    """
    WiFi altyazı alıcı — PC'den gelen sonuçları dinler.

    Parameters
    ----------
    port : int
        Dinleme portu.
    bind_ip : str
        Dinleme adresi. "0.0.0.0" = tüm arayüzler.
    """

    def __init__(self, port: int = 8555, bind_ip: str = "0.0.0.0"):
        self.port = port
        self.bind_ip = bind_ip

        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._running = False
        self._thread = None
        self._connected = False

        # Callback'ler
        self.on_subtitle: Optional[Callable[[str, float], None]] = None
        self.on_expression: Optional[Callable[[str], None]] = None
        self.on_connection_change: Optional[Callable[[bool], None]] = None

        # İstatistikler
        self.messages_received = 0
        self.last_subtitle = ""
        self.last_confidence = 0.0
        self.last_expression = ""

    def start(self):
        """Altyazı sunucusunu başlat ve bağlantı bekle."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="subtitle-receiver"
        )
        self._thread.start()
        logger.info(f"SubtitleReceiver dinliyor: {self.bind_ip}:{self.port}")

    def stop(self):
        """Sunucuyu durdur."""
        self._running = False
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("SubtitleReceiver durduruldu")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _listen_loop(self):
        """Ana dinleme döngüsü — bağlantı kabul et, mesaj oku."""
        while self._running:
            try:
                # Sunucu socket oluştur
                self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )
                self._server_socket.settimeout(2.0)
                self._server_socket.bind((self.bind_ip, self.port))
                self._server_socket.listen(1)

                logger.info(f"Bağlantı bekleniyor: port {self.port}")

                while self._running:
                    try:
                        client, addr = self._server_socket.accept()
                        self._client_socket = client
                        self._connected = True
                        logger.info(f"PC bağlandı: {addr}")

                        if self.on_connection_change:
                            self.on_connection_change(True)

                        self._receive_messages()

                    except socket.timeout:
                        continue
                    except OSError:
                        if self._running:
                            break

            except Exception as e:
                logger.error(f"SubtitleReceiver hatası: {e}")
                time.sleep(2.0)
            finally:
                self._connected = False
                if self.on_connection_change:
                    self.on_connection_change(False)
                if self._server_socket:
                    try:
                        self._server_socket.close()
                    except Exception:
                        pass

    def _receive_messages(self):
        """Bağlı client'tan mesaj al."""
        while self._running and self._connected:
            try:
                # 4 byte boyut header oku
                header = self._recv_exact(4)
                if header is None:
                    break

                msg_size = struct.unpack(">I", header)[0]
                if msg_size > 65536:  # 64KB güvenlik limiti
                    logger.warning(f"Çok büyük mesaj: {msg_size} bytes — atlanıyor")
                    break

                # Mesaj gövdesi oku
                data = self._recv_exact(msg_size)
                if data is None:
                    break

                # JSON parse
                msg = json.loads(data.decode("utf-8"))
                self._handle_message(msg)

            except (ConnectionResetError, BrokenPipeError):
                logger.info("PC bağlantısı koptu")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse hatası: {e}")
            except Exception as e:
                logger.error(f"Mesaj okuma hatası: {e}")
                break

        self._connected = False
        if self.on_connection_change:
            self.on_connection_change(False)

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Tam n byte oku."""
        data = b""
        while len(data) < n:
            try:
                chunk = self._client_socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None
        return data

    def _handle_message(self, msg: dict):
        """Gelen mesajı işle."""
        self.messages_received += 1
        msg_type = msg.get("type", "subtitle")

        if msg_type == "subtitle":
            text = msg.get("text", "")
            confidence = msg.get("confidence", 0.0)
            self.last_subtitle = text
            self.last_confidence = confidence

            if self.on_subtitle:
                self.on_subtitle(text, confidence)

            logger.debug(f"Altyazı: '{text}' (güven: {confidence:.2f})")

        elif msg_type == "expression":
            expression = msg.get("expression", "")
            self.last_expression = expression

            if self.on_expression:
                self.on_expression(expression)

        elif msg_type == "status":
            logger.info(f"PC durum: {msg.get('message', '')}")
