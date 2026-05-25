"""
backend/gpio_alert.py
─────────────────────
Raspberry Pi Zero GPIO üzerinden fiziksel uyarı sistemi.

Bilişsel yük seviyesine ve dudak okuma tahminlerine göre:
- LED (RGB): Yeşil (düşük yük) → Sarı (orta) → Kırmızı (yüksek)
- Buzzer: Tehlike seviyesinde kısa bip
- Titreşim: Kelime tahmin onayı

GPIO Pinleri (BCM numaralandırma):
    LED_GREEN  = GPIO 17
    LED_YELLOW = GPIO 27
    LED_RED    = GPIO 22
    BUZZER     = GPIO 23
    VIBRATION  = GPIO 24

Kullanım:
    from backend.gpio_alert import GPIOAlert

    alert = GPIOAlert()
    alert.set_cognitive_level(0.7)      # %70 bilişsel yük → sarı LED
    alert.confirm_prediction("merhaba") # Titreşim onayı
    alert.cleanup()
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# GPIO pin tanımları (BCM numaralandırma)
PIN_LED_GREEN = 17
PIN_LED_YELLOW = 27
PIN_LED_RED = 22
PIN_BUZZER = 23
PIN_VIBRATION = 24

# Bilişsel yük eşikleri
THRESHOLD_LOW = 0.3
THRESHOLD_MEDIUM = 0.6
THRESHOLD_HIGH = 0.85


class GPIOAlert:
    """Pi Zero GPIO tabanlı fiziksel uyarı sistemi.

    Mock modda çalışabilir (PC'de GPIO olmadan test için).
    """

    def __init__(self, mock: bool = False):
        """
        Args:
            mock: True ise GPIO'ya erişmez, sadece log yazar
        """
        self.mock = mock
        self._gpio = None
        self._available = False
        self._current_level = "low"

        if not mock:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)

                # Pin çıkışlarını ayarla
                for pin in [PIN_LED_GREEN, PIN_LED_YELLOW, PIN_LED_RED, PIN_BUZZER, PIN_VIBRATION]:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)

                self._available = True
                logger.info("GPIO Alert sistemi hazır (BCM modu)")

                # Başlangıçta yeşil LED
                GPIO.output(PIN_LED_GREEN, GPIO.HIGH)

            except ImportError:
                logger.info("RPi.GPIO bulunamadı — mock modda çalışıyor")
                self.mock = True
            except Exception as e:
                logger.warning(f"GPIO başlatılamadı: {e} — mock modda çalışıyor")
                self.mock = True
        else:
            logger.info("GPIO Alert mock modda başlatıldı")

    def set_cognitive_level(self, level: float):
        """Bilişsel yük seviyesine göre LED'leri güncelle.

        Args:
            level: 0.0 (düşük yük) - 1.0 (yüksek yük)
        """
        level = max(0.0, min(1.0, level))

        if level < THRESHOLD_LOW:
            new_level = "low"
        elif level < THRESHOLD_MEDIUM:
            new_level = "medium"
        elif level < THRESHOLD_HIGH:
            new_level = "high"
        else:
            new_level = "critical"

        if new_level == self._current_level:
            return

        self._current_level = new_level

        if self.mock:
            emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
            logger.debug(f"GPIO: {emoji.get(new_level, '⚪')} Bilişsel yük: {level:.2f} ({new_level})")
            return

        GPIO = self._gpio

        # Tüm LED'leri kapat
        GPIO.output(PIN_LED_GREEN, GPIO.LOW)
        GPIO.output(PIN_LED_YELLOW, GPIO.LOW)
        GPIO.output(PIN_LED_RED, GPIO.LOW)

        if new_level == "low":
            GPIO.output(PIN_LED_GREEN, GPIO.HIGH)
        elif new_level == "medium":
            GPIO.output(PIN_LED_YELLOW, GPIO.HIGH)
        elif new_level == "high":
            GPIO.output(PIN_LED_RED, GPIO.HIGH)
        elif new_level == "critical":
            GPIO.output(PIN_LED_RED, GPIO.HIGH)
            # Buzzer uyarısı (arka planda)
            threading.Thread(target=self._buzz, args=(0.3, 2), daemon=True).start()

    def confirm_prediction(self, word: str):
        """Kelime tahmini onaylandığında titreşim ver.

        Args:
            word: Tahmin edilen kelime
        """
        if self.mock:
            logger.debug(f"GPIO: 📳 Tahmin onayı: '{word}'")
            return

        threading.Thread(target=self._vibrate, args=(0.15,), daemon=True).start()

    def _buzz(self, duration: float = 0.2, count: int = 1):
        """Buzzer çal."""
        if not self._available:
            return
        GPIO = self._gpio
        for i in range(count):
            GPIO.output(PIN_BUZZER, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(PIN_BUZZER, GPIO.LOW)
            if i < count - 1:
                time.sleep(0.1)

    def _vibrate(self, duration: float = 0.15):
        """Titreşim motoru çalıştır."""
        if not self._available:
            return
        GPIO = self._gpio
        GPIO.output(PIN_VIBRATION, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(PIN_VIBRATION, GPIO.LOW)

    def cleanup(self):
        """GPIO pinlerini temizle."""
        if self._available and self._gpio:
            for pin in [PIN_LED_GREEN, PIN_LED_YELLOW, PIN_LED_RED, PIN_BUZZER, PIN_VIBRATION]:
                self._gpio.output(pin, self._gpio.LOW)
            self._gpio.cleanup()
            logger.info("GPIO temizlendi")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
