"""
pi/mqtt_subtitle_rx.py
══════════════════════
MQTT Altyazı Alıcı — Raspberry Pi 3 B+ Tarafı

PC'den MQTT üzerinden gelen altyazı mesajlarını alır
ve OLED ekrana iletilmek üzere callback'e verir.

Topic: blindeye/subtitle
Format: {"line1": "...", "line2": "...", "confidence": 0.85, ...}

Kullanım:
    def on_subtitle(line1, line2, confidence):
        oled.show(line1, line2)

    rx = MQTTSubtitleReceiver(broker_host="192.168.1.100")
    rx.on_subtitle = on_subtitle
    rx.start()
    ...
    rx.stop()
"""

import json
import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    logger.warning("paho-mqtt yüklü değil — MQTT alıcı çalışmayacak.")


class MQTTSubtitleReceiver:
    """Pi tarafı MQTT altyazı alıcı.

    Parameters
    ----------
    broker_host : str
        PC'nin MQTT broker IP adresi.
    broker_port : int
        MQTT broker portu.
    client_id : str
        Pi MQTT client ID.
    topic_subtitle : str
        Altyazı topic'i.
    topic_status : str
        Durum topic'i.
    topic_heartbeat : str
        Heartbeat topic'i (Pi → PC).
    heartbeat_interval_s : float
        Heartbeat gönderim aralığı.
    """

    def __init__(
        self,
        broker_host: str = "192.168.1.100",
        broker_port: int = 1883,
        client_id: str = "blindeye-pi",
        topic_subtitle: str = "blindeye/subtitle",
        topic_status: str = "blindeye/status",
        topic_heartbeat: str = "blindeye/heartbeat",
        heartbeat_interval_s: float = 5.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.topic_subtitle = topic_subtitle
        self.topic_status = topic_status
        self.topic_heartbeat = topic_heartbeat
        self.heartbeat_interval_s = heartbeat_interval_s

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Callback: (line1: str, line2: str, confidence: float) → None
        self.on_subtitle: Optional[Callable] = None

        # İstatistikler
        self.messages_received: int = 0
        self.last_subtitle_time: float = 0.0

    def start(self):
        """MQTT alıcıyı başlat."""
        if not _MQTT_AVAILABLE:
            logger.error("paho-mqtt gerekli — pip install paho-mqtt")
            return

        if self._running:
            return

        self._running = True

        try:
            self._client = mqtt.Client(
                client_id=self.client_id,
                protocol=mqtt.MQTTv311,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # LWT
            lwt = json.dumps({"status": "offline", "device": "pi"})
            self._client.will_set(self.topic_status, lwt, qos=1, retain=True)

            # Bağlan
            self._client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()

            # Heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="mqtt-heartbeat"
            )
            self._heartbeat_thread.start()

            logger.info(f"MQTT alıcı başlatılıyor → {self.broker_host}:{self.broker_port}")

        except Exception as e:
            logger.error(f"MQTT başlatma hatası: {e}")

    def stop(self):
        """MQTT alıcıyı durdur."""
        self._running = False

        if self._client is not None:
            try:
                self._client.publish(
                    self.topic_status,
                    json.dumps({"status": "offline", "device": "pi"}),
                    qos=1,
                    retain=True,
                )
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

        logger.info(f"MQTT alıcı durduruldu — {self.messages_received} mesaj alındı")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ──────────── CALLBACKS ────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected = True
            client.subscribe(self.topic_subtitle, qos=1)
            client.subscribe(self.topic_status, qos=0)

            # Online durumu bildir
            client.publish(
                self.topic_status,
                json.dumps({"status": "online", "device": "pi"}),
                qos=1,
                retain=True,
            )
            logger.info(f"✅ MQTT broker'a bağlandı — altyazı dinleniyor")
        else:
            logger.warning(f"MQTT bağlantı reddedildi: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = False
        if reason_code != 0:
            logger.warning(f"MQTT bağlantı koptu: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """Gelen MQTT mesajını işle."""
        try:
            if msg.topic == self.topic_subtitle:
                data = json.loads(msg.payload.decode("utf-8"))
                line1 = data.get("line1", "")
                line2 = data.get("line2", "")
                confidence = data.get("confidence", 0.0)

                self.messages_received += 1
                self.last_subtitle_time = time.time()

                # Callback ile OLED'e ilet
                if self.on_subtitle is not None:
                    self.on_subtitle(line1, line2, confidence)

        except json.JSONDecodeError:
            logger.warning(f"JSON parse hatası: {msg.payload[:50]}")
        except Exception as e:
            logger.warning(f"Mesaj işleme hatası: {e}")

    def _heartbeat_loop(self):
        """Periyodik heartbeat gönder (Pi → PC)."""
        while self._running:
            if self._connected:
                try:
                    payload = json.dumps({
                        "device": "pi",
                        "uptime_s": time.monotonic(),
                        "msgs_rx": self.messages_received,
                        "timestamp": time.time(),
                    })
                    self._client.publish(
                        self.topic_heartbeat, payload, qos=0
                    )
                except Exception:
                    pass
            time.sleep(self.heartbeat_interval_s)
