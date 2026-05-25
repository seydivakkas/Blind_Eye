from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QLinearGradient
from PyQt6.QtCore import Qt, QRectF
import cv2
import numpy as np
import time

from .styles import Theme


class VideoWidget(QLabel):
    """Kamera görüntüsü + ROI overlay + FPS counter + durum göstergesi."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._has_frame = False
        self._fps_counter = 0
        self._fps_display = 0.0
        self._last_fps_time = time.time()
        self._frame_count = 0
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {Theme.BG_DEEP};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_LG};
                color: {Theme.TEXT_MUTED};
                font-size: 16px;
                font-weight: 500;
            }}
        """)
        self.setText("Kamera Bekleniyor...")

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        self._has_frame = True
        self._frame_count += 1

        # FPS hesaplama
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps_display = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(img).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._has_frame:
            # ── FPS Counter (sol üst) ──
            fps_text = f"{self._fps_display:.0f} FPS"
            font = QFont(Theme.FONT_MONO.split(",")[0].strip("' "), 10)
            font.setBold(True)
            painter.setFont(font)

            # Arka plan kutusu
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(fps_text)
            text_height = metrics.height()
            padding = 6
            rect = QRectF(8, 8, text_width + padding * 2, text_height + padding)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(rect, 6, 6)

            # Metin
            color = QColor(Theme.SUCCESS) if self._fps_display >= 24 else \
                    QColor(Theme.WARNING) if self._fps_display >= 15 else \
                    QColor(Theme.ERROR)
            painter.setPen(color)
            painter.drawText(
                int(rect.x() + padding),
                int(rect.y() + text_height),
                fps_text,
            )
        else:
            # ── Idle State: Kamera ikonu ──
            cx, cy = self.width() // 2, self.height() // 2 - 20

            # Dış daire (glow efekti)
            glow = QColor(Theme.ACCENT)
            glow.setAlpha(30)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(cx - 36, cy - 36, 72, 72)

            # Dış halka
            pen = QPen(QColor(Theme.TEXT_MUTED), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(cx - 24, cy - 24, 48, 48)

            # İç daire (accent)
            painter.setBrush(QColor(Theme.ACCENT))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 8, cy - 8, 16, 16)

            # Alt metin
            painter.setPen(QColor(Theme.TEXT_MUTED))
            font = QFont(Theme.FONT_FAMILY.split(",")[0].strip("' "), 11)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(0, 30, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "Başlat'a basarak kamerayı açın",
            )

        painter.end()
