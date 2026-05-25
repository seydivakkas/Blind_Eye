"""
pc/oled_sender.py
═════════════════
MQTT ile Raspberry Pi 3 B+'ya Altyazı Gönderici

PC'de çalışan VSR pipeline'ın çıktısını (altyazı metni)
MQTT protokolü üzerinden Pi'ye iletir.

Topic yapısı:
    blindeye/subtitle  — PC → Pi: altyazı JSON
    blindeye/status    — çift yönlü durum mesajları
    blindeye/heartbeat — Pi → PC: yaşam kontrolü

Mesaj formatı (blindeye/subtitle):
    {
        "line1": "merhaba dünya",
        "line2": "nasılsınız",
        "confidence": 0.85,
        "punctuation": ".",
        "timestamp": 1716571234.5
    }

Kullanım:
    sender = OledSender(broker_host="192.168.1.100")
    sender.start()
    sender.send_subtitle("merhaba dünya", confidence=0.85)
    sender.stop()
"""

import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    logger.warning(
        "paho-mqtt yüklü değil — OledSender mock modda çalışacak. "
        "pip install paho-mqtt"
    )


class OledSender:
    """MQTT altyazı gönderici — PC → Pi.

    Parameters
    ----------
    broker_host : str
        MQTT broker IP adresi (genelde PC'nin kendi IP'si).
    broker_port : int
        MQTT broker portu.
    qos : int
        Quality of Service. 1 = en az bir kez teslim.
    client_id : str
        MQTT client tanımlayıcı.
    topic_subtitle : str
        Altyazı topic'i.
    topic_status : str
        Durum topic'i.
    max_rate_hz : float
        Maksimum gönderim hızı (Hz). OLED refresh rate'e uygun.
    reconnect_interval_s : float
        Bağlantı koparsa yeniden deneme aralığı.
    """

    def __init__(
        self,
        broker_host: str = "192.168.1.100",
        broker_port: int = 1883,
        qos: int = 1,
        client_id: str = "blindeye-pc",
        topic_subtitle: str = "blindeye/subtitle",
        topic_status: str = "blindeye/status",
        max_rate_hz: float = 5.0,
        reconnect_interval_s: float = 3.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.qos = qos
        self.client_id = client_id
        self.topic_subtitle = topic_subtitle
        self.topic_status = topic_status
        self.reconnect_interval_s = reconnect_interval_s

        self._min_interval_s = 1.0 / max_rate_hz
        self._last_send_time: float = 0.0

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._running = False
        self._mock = not _MQTT_AVAILABLE

        # İstatistikler
        self.messages_sent: int = 0
        self.errors: int = 0

    # ──────────── PUBLIC API ────────────

    def start(self):
        """MQTT bağlantısını başlat."""
        if self._running:
            return
        self._running = True

        if self._mock:
            logger.info("OledSender: Mock mod aktif (paho-mqtt yok)")
            return

        try:
            self._client = mqtt.Client(
                client_id=self.client_id,
                protocol=mqtt.MQTTv311,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )

            # Callback'ler
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            # Last Will & Testament
            lwt_payload = json.dumps(
                {"status": "offline", "device": "pc"},
                ensure_ascii=False,
            )
            self._client.will_set(
                self.topic_status, lwt_payload, qos=self.qos, retain=True
            )

            # Bağlan (non-blocking)
            self._client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()

            logger.info(
                f"OledSender başlatılıyor → {self.broker_host}:{self.broker_port}"
            )

        except Exception as e:
            logger.error(f"MQTT bağlantı hatası: {e}")
            self._mock = True

    def stop(self):
        """MQTT bağlantısını kapat."""
        self._running = False

        if self._client is not None:
            try:
                # Offline durumu bildir
                self._client.publish(
                    self.topic_status,
                    json.dumps({"status": "offline", "device": "pc"}),
                    qos=self.qos,
                    retain=True,
                )
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

        logger.info(f"OledSender durduruldu — {self.messages_sent} mesaj gönderildi")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_subtitle(
        self,
        line1: str,
        line2: str = "",
        confidence: float = 0.0,
        punctuation: str = "",
    ):
        """Altyazı metnini Pi'ye gönder.

        Parameters
        ----------
        line1 : str
            Üst satır (son cümle/kelime).
        line2 : str
            Alt satır (önceki cümle).
        confidence : float
            VSR güven skoru.
        punctuation : str
            Eklenen punctuation karakteri.
        """
        # Rate limiting
        now = time.time()
        if now - self._last_send_time < self._min_interval_s:
            return
        self._last_send_time = now

        payload = {
            "line1": line1,
            "line2": line2,
            "confidence": round(confidence, 3),
            "punctuation": punctuation,
            "timestamp": now,
        }

        if self._mock:
            logger.info(f"📺 MQTT→Pi: \"{line1}\" | \"{line2}\" ({confidence:.0%})")
            self.messages_sent += 1
            return

        if not self._connected:
            return

        try:
            msg = json.dumps(payload, ensure_ascii=False)
            self._client.publish(
                self.topic_subtitle, msg, qos=self.qos
            )
            self.messages_sent += 1
        except Exception as e:
            logger.warning(f"MQTT gönderim hatası: {e}")
            self.errors += 1

    def send_status(self, message: str):
        """Durum mesajı gönder."""
        payload = {
            "status": "online",
            "device": "pc",
            "message": message,
            "timestamp": time.time(),
        }

        if self._mock:
            logger.info(f"📡 MQTT status: {message}")
            return

        if not self._connected:
            return

        try:
            msg = json.dumps(payload, ensure_ascii=False)
            self._client.publish(self.topic_status, msg, qos=self.qos, retain=True)
        except Exception as e:
            logger.debug(f"Status gönderim hatası: {e}")

    # ──────────── CALLBACKS ────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """MQTT broker'a bağlandığında."""
        if reason_code == 0:
            self._connected = True
            logger.info(f"✅ MQTT broker'a bağlandı: {self.broker_host}")
            # Online durumu bildir
            self.send_status("Pipeline çalışıyor")
        else:
            logger.warning(f"MQTT bağlantı reddedildi: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        """MQTT bağlantısı koptuğunda."""
        self._connected = False
        if reason_code != 0:
            logger.warning(f"MQTT bağlantı koptu: {reason_code}")
