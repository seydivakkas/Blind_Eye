from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import Qt
from .styles import Theme
import time


class SubtitleView(QTextEdit):
    """Gerçek zamanlı altyazı gösterimi — zaman damgası + confidence renk kodlaması."""

    MAX_ENTRIES = 50  # Bellek yönetimi: en fazla 50 giriş tut

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(180)
        self._entry_count = 0
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.BG_DEEP};
                color: {Theme.TEXT_PRIMARY};
                font-size: 15px;
                font-family: {Theme.FONT_FAMILY};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD};
                padding: 12px;
                line-height: 1.6;
            }}
        """)
        self.setPlaceholderText("Altyazılar burada görünecek...")

    def append_text(self, text: str, conf: float):
        if not text.strip():
            return

        # Bellek yönetimi: çok uzarsa en eski girişleri sil
        self._entry_count += 1
        if self._entry_count > self.MAX_ENTRIES:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 3)
            cursor.removeSelectedText()
            self._entry_count -= 1

        # Zaman damgası
        timestamp = time.strftime("%H:%M:%S")

        color = Theme.confidence_color(conf)
        badge_bg = f"rgba({self._hex_to_rgb(color)}, 0.15)"

        html = (
            f'<div style="margin-bottom: 6px;">'
            f'<span style="'
            f'  color: {Theme.TEXT_MUTED};'
            f'  font-size: 10px;'
            f'  font-family: {Theme.FONT_MONO};'
            f'">{timestamp}</span> '
            f'<span style="'
            f'  background-color: {badge_bg};'
            f'  color: {color};'
            f'  padding: 2px 6px;'
            f'  border-radius: 4px;'
            f'  font-size: 11px;'
            f'  font-family: {Theme.FONT_MONO};'
            f'">{conf:.0%}</span> '
            f'<span style="color: {Theme.TEXT_PRIMARY}; font-size: 15px;">{text}</span>'
            f'</div>'
        )
        self.append(html)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r}, {g}, {b}"

    def clear_subtitles(self):
        """Tüm altyazıları temizler."""
        self.clear()
        self._entry_count = 0
