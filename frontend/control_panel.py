from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox,
    QLabel, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal
from .styles import Theme


class ControlPanel(QWidget):
    """Kontrol butonları + chunk size ayarı + status dot + log toggle."""

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    chunk_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Theme.BG_CARD};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_LG};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # ── Başlık ──
        title = QLabel("Kontroller")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Theme.TEXT_SECONDARY}; "
            f"letter-spacing: 1px; text-transform: uppercase; border: none;"
        )
        layout.addWidget(title)

        # ── Butonlar ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_start = QPushButton("▶  Başlat")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setToolTip("Pipeline'ı başlat ve kamerayı aç")
        self.btn_start.clicked.connect(self._on_start)

        self.btn_stop = QPushButton("■  Durdur")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Pipeline'ı güvenli şekilde durdur")
        self.btn_stop.clicked.connect(self._on_stop)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # ── Ayarlar ──
        settings_row = QHBoxLayout()
        settings_row.setSpacing(12)

        chunk_label = QLabel("Chunk:")
        chunk_label.setStyleSheet(
            f"color: {Theme.TEXT_SECONDARY}; font-size: 12px; border: none;"
        )
        chunk_label.setToolTip("Kaç frame bir araya gelince inference çalışır")
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(3, 12)
        self.chunk_spin.setValue(6)
        self.chunk_spin.setFixedWidth(60)
        self.chunk_spin.valueChanged.connect(self.chunk_changed.emit)

        self.log_check = QCheckBox("CSV Log")
        self.log_check.setChecked(True)
        self.log_check.setStyleSheet(f"border: none;")
        self.log_check.setToolTip("Metrik verilerini CSV dosyasına kaydet")

        settings_row.addWidget(chunk_label)
        settings_row.addWidget(self.chunk_spin)
        settings_row.addStretch()
        settings_row.addWidget(self.log_check)
        layout.addLayout(settings_row)

        # ── Durum göstergesi (dot + label) ──
        status_row = QHBoxLayout()
        status_row.setSpacing(8)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"font-size: 10px; color: {Theme.TEXT_MUTED}; border: none;"
        )

        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet(
            f"color: {Theme.TEXT_MUTED}; font-size: 11px; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )

        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

    def _on_start(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.chunk_spin.setEnabled(False)
        self._set_dot_color(Theme.SUCCESS)
        self.start_clicked.emit()

    def _on_stop(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.chunk_spin.setEnabled(True)
        self._set_dot_color(Theme.TEXT_MUTED)
        self.stop_clicked.emit()

    def set_status(self, text: str):
        self.status_label.setText(text)
        # Durum metnine göre dot rengi
        text_lower = text.lower()
        if "çalışıyor" in text_lower or "bağlı" in text_lower:
            self._set_dot_color(Theme.SUCCESS)
        elif "mock" in text_lower or "bekl" in text_lower:
            self._set_dot_color(Theme.WARNING)
        elif "hata" in text_lower or "durdur" in text_lower:
            self._set_dot_color(Theme.ERROR)
        else:
            self._set_dot_color(Theme.TEXT_MUTED)

    def _set_dot_color(self, color: str):
        self.status_dot.setStyleSheet(
            f"font-size: 10px; color: {color}; border: none;"
        )

    def reset(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.chunk_spin.setEnabled(True)
        self._set_dot_color(Theme.TEXT_MUTED)
        self.status_label.setText("Hazır")
