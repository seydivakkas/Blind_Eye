"""
Blind Eye — SSD1306 OLED Display (Pi Gözlük HUD)
=================================================
128×64 OLED ekranda altyazı gösterimi.

Kullanım:
    oled = OledDisplay(width=128, height=64)
    oled.start()
    oled.show_subtitle("merhaba dünya")
    oled.show_status(connected=True, fps=20)
    oled.stop()
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports — Pi dışındaki platformlarda luma olmayabilir
_luma_available = False
_PIL_available = False


def _check_luma():
    """luma.oled kütüphanesini kontrol et."""
    global _luma_available
    try:
        from luma.core.interface.serial import i2c  # noqa: F401
        from luma.oled.device import ssd1306  # noqa: F401
        _luma_available = True
    except ImportError:
        _luma_available = False
    return _luma_available


def _check_pil():
    """PIL kütüphanesini kontrol et."""
    global _PIL_available
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        _PIL_available = True
    except ImportError:
        _PIL_available = False
    return _PIL_available


class OledDisplay:
    """
    SSD1306 OLED HUD kontrolcüsü.

    luma.oled yoksa mock mod — log'a yazar.

    Parameters
    ----------
    width : int
        Ekran genişliği (piksel). Genellikle 128.
    height : int
        Ekran yüksekliği (piksel). Genellikle 64.
    i2c_address : int
        I2C adresi. Genellikle 0x3C.
    i2c_port : int
        I2C port numarası. Genellikle 1.
    font_size : int
        Yazı tipi boyutu (piksel).
    contrast : int
        Kontrast (0–255).
    rotate : int
        Döndürme (0 veya 2=180°).
    max_lines : int
        Ekranda gösterilecek max satır sayısı.
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 64,
        i2c_address: int = 0x3C,
        i2c_port: int = 1,
        font_size: int = 12,
        contrast: int = 200,
        rotate: int = 0,
        max_lines: int = 4,
    ):
        self.width = width
        self.height = height
        self.i2c_address = i2c_address
        self.i2c_port = i2c_port
        self.font_size = font_size
        self.contrast = contrast
        self.rotate = rotate
        self.max_lines = max_lines

        self._device = None
        self._font = None
        self._small_font = None
        self._running = False
        self._lock = threading.Lock()
        self._mock = False

        # Ekran durumu
        self._current_lines: list[str] = []
        self._status_text = ""
        self._connected = False
        self._expression = ""

    def start(self):
        """OLED ekranı başlat."""
        if self._running:
            return

        if _check_luma() and _check_pil():
            self._init_hardware()
        else:
            logger.warning(
                "luma.oled veya PIL bulunamadı — OLED MOCK mod aktif. "
                "pip install luma.oled pillow"
            )
            self._mock = True

        self._running = True
        self._clear()
        self._show_splash()
        logger.info(f"OledDisplay başlatıldı: {self.width}×{self.height}, "
                     f"mock={self._mock}")

    def stop(self):
        """OLED ekranı durdur."""
        self._running = False
        if self._device and not self._mock:
            try:
                self._clear()
                self._device.hide()
            except Exception:
                pass
        logger.info("OledDisplay durduruldu")

    def show_subtitle(self, text: str, confidence: float = 0.0):
        """
        Altyazı metnini göster.

        Uzun metin otomatik olarak satırlara bölünür
        ve ekran taşarsa eski satırlar kaldırılır.
        """
        if not text.strip():
            return

        with self._lock:
            # Metni satırlara böl (word-wrap)
            wrapped = self._word_wrap(text, self.width - 4)
            self._current_lines.extend(wrapped)

            # Max satır sınırı
            if len(self._current_lines) > self.max_lines:
                self._current_lines = self._current_lines[-self.max_lines:]

            self._render()

        if self._mock:
            conf_str = f" [{confidence:.0%}]" if confidence > 0 else ""
            logger.info(f"📺 OLED: {text}{conf_str}")

    def show_expression(self, expression: str):
        """Mimik durumunu göster (üst köşe ikonu)."""
        self._expression = expression
        self._render()

    def show_status(self, connected: bool = False, fps: float = 0.0):
        """Durum satırını güncelle (alt kısım)."""
        wifi_icon = "📶" if connected else "❌"
        self._status_text = f"{wifi_icon} {fps:.0f}fps"
        self._connected = connected
        self._render()

    def clear(self):
        """Ekranı temizle."""
        with self._lock:
            self._current_lines.clear()
            self._expression = ""
            self._status_text = ""
            self._clear()

    # ──────────── INTERNAL ────────────

    def _init_hardware(self):
        """Gerçek OLED donanımını başlat."""
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306

            serial = i2c(port=self.i2c_port, address=self.i2c_address)
            self._device = ssd1306(
                serial,
                width=self.width,
                height=self.height,
                rotate=self.rotate,
            )
            self._device.contrast(self.contrast)

            # Font yükleme
            from PIL import ImageFont
            try:
                self._font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    self.font_size,
                )
                self._small_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    max(8, self.font_size - 4),
                )
            except (OSError, IOError):
                self._font = ImageFont.load_default()
                self._small_font = self._font

            logger.info("OLED donanımı başlatıldı")
        except Exception as e:
            logger.warning(f"OLED donanımı başlatılamadı: {e} — mock moda geçiliyor")
            self._mock = True

    def _render(self):
        """Ekranı yeniden çiz."""
        if self._mock or self._device is None:
            return

        try:
            from PIL import Image, ImageDraw

            img = Image.new("1", (self.width, self.height), 0)
            draw = ImageDraw.Draw(img)

            y = 0
            line_height = self.font_size + 2

            # Durum satırı (üstte, küçük font)
            if self._status_text:
                draw.text((0, 0), self._status_text, fill=1, font=self._small_font)
                y = 10

            # Mimik ikonu (sağ üst)
            if self._expression:
                expr_icons = {
                    "mutlu": ":)",
                    "üzgün": ":(",
                    "şaşkın": ":O",
                    "kızgın": ">:",
                    "nötr": ":-",
                }
                icon = expr_icons.get(self._expression, "?")
                draw.text(
                    (self.width - 20, 0), icon, fill=1, font=self._small_font
                )

            # Altyazı satırları
            for line in self._current_lines:
                if y + line_height > self.height:
                    break
                draw.text((2, y), line, fill=1, font=self._font)
                y += line_height

            self._device.display(img)
        except Exception as e:
            logger.debug(f"OLED render hatası: {e}")

    def _clear(self):
        """Ekranı siyah yap."""
        if self._mock or self._device is None:
            return
        try:
            from PIL import Image
            img = Image.new("1", (self.width, self.height), 0)
            self._device.display(img)
        except Exception:
            pass

    def _show_splash(self):
        """Başlangıç splash ekranı."""
        if self._mock:
            logger.info("📺 OLED: [Blind Eye v2.0]")
            return

        if self._device is None:
            return

        try:
            from PIL import Image, ImageDraw

            img = Image.new("1", (self.width, self.height), 0)
            draw = ImageDraw.Draw(img)

            draw.text((20, 10), "Blind Eye", fill=1, font=self._font)
            draw.text((30, 30), "v2.0", fill=1, font=self._small_font or self._font)
            draw.text((10, 48), "WiFi bekleniyor...", fill=1, font=self._small_font or self._font)

            self._device.display(img)
        except Exception:
            pass

    def _word_wrap(self, text: str, max_width: int) -> list[str]:
        """Metni ekran genişliğine göre satırlara böl."""
        if self._mock:
            # Mock modda basit bölme
            words = text.split()
            lines = []
            current = ""
            chars_per_line = max_width // (self.font_size // 2 + 1)
            for word in words:
                test = f"{current} {word}".strip()
                if len(test) > chars_per_line:
                    if current:
                        lines.append(current)
                    current = word
                else:
                    current = test
            if current:
                lines.append(current)
            return lines or [text]

        # PIL ile gerçek piksel genişliği hesabı
        try:
            from PIL import Image, ImageDraw

            img = Image.new("1", (1, 1))
            draw = ImageDraw.Draw(img)
            font = self._font

            words = text.split()
            lines = []
            current = ""

            for word in words:
                test = f"{current} {word}".strip()
                bbox = draw.textbbox((0, 0), test, font=font)
                text_width = bbox[2] - bbox[0]

                if text_width > max_width and current:
                    lines.append(current)
                    current = word
                else:
                    current = test

            if current:
                lines.append(current)

            return lines or [text]
        except Exception:
            return [text]
