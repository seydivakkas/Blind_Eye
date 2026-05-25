from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QCloseEvent, QKeySequence, QShortcut

from backend.pipeline import PipelineController
from .video_widget import VideoWidget
from .subtitle_view import SubtitleView
from .metrics_panel import MetricsPanel
from .control_panel import ControlPanel
from .styles import Theme, GLOBAL_QSS

import logging
import time

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Premium PyQt6 ana pencere — layout, signal binding, lifecycle."""

    def __init__(self, model_path: str = "models/student_int8.onnx",
                 chunk_size: int = 6):
        super().__init__()

        self.setWindowTitle("Blind Eye — Türkçe Dudak Okuma Prototipi")
        self.resize(1100, 700)
        self.setMinimumSize(900, 550)
        self.setStyleSheet(GLOBAL_QSS)

        self.pipeline = PipelineController(
            model_path=model_path, chunk_size=chunk_size
        )

        self._subtitle_font_size = 15  # Erişilebilirlik: ayarlanabilir font

        self._setup_ui()
        self._bind_signals()
        self._setup_timers()
        self._setup_shortcuts()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ═══ SOL PANEL: Video + Altyazı ═══
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Logo / Başlık
        header = QHBoxLayout()
        logo = QLabel("◉ Blind Eye")
        logo.setStyleSheet(
            f"font-size: 18px; font-weight: 800; color: {Theme.ACCENT}; "
            f"letter-spacing: -0.5px;"
        )
        version = QLabel("v1.0 — TÜBİTAK 2209-A")
        version.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_MUTED};"
        )
        header.addWidget(logo)
        header.addStretch()
        header.addWidget(version)
        left_layout.addLayout(header)

        # Video widget
        self.video = VideoWidget()
        left_layout.addWidget(self.video, stretch=3)

        # Altyazı
        sub_label = QLabel("Altyazı Çıktısı")
        sub_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {Theme.TEXT_SECONDARY}; "
            f"padding: 4px 0 0 2px;"
        )
        left_layout.addWidget(sub_label)

        self.subtitle = SubtitleView()
        left_layout.addWidget(self.subtitle, stretch=1)

        # ═══ SAĞ PANEL: Metrikler + Kontroller ═══
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Sağ panel başlık
        right_header = QLabel("Dashboard")
        right_header.setStyleSheet(
            f"font-size: 18px; font-weight: 800; color: {Theme.TEXT_PRIMARY}; "
            f"padding: 0 0 4px 0;"
        )
        right_layout.addWidget(right_header)

        # Metrikler
        self.metrics = MetricsPanel()
        right_layout.addWidget(self.metrics)

        # Ayırıcı
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {Theme.BORDER};")
        right_layout.addWidget(separator)

        # Kontroller
        self.controls = ControlPanel()
        right_layout.addWidget(self.controls)

        right_layout.addStretch()

        # ═══ SPLITTER ═══
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([680, 380])
        splitter.setHandleWidth(3)

        main_layout.addWidget(splitter)

        # ═══ STATUS BAR ═══
        status = QStatusBar()
        self.setStatusBar(status)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 10px;")
        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet(f"color: {Theme.TEXT_MUTED};")

        status.addWidget(self.status_dot)
        status.addWidget(self.status_label)

        # Sağ taraf: KVKK notu
        kvkk_label = QLabel("🔒 RAM-only · Disk kaydı yok")
        kvkk_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 10px;")
        kvkk_label.setToolTip("KVKK uyumlu: Video disk'e kaydedilmez, sadece RAM'de işlenir.")
        status.addPermanentWidget(kvkk_label)

    def _bind_signals(self):
        # Pipeline → UI
        self.pipeline.frame_ready.connect(self.video.update_frame)
        self.pipeline.subtitle_ready.connect(self.subtitle.append_text)
        self.pipeline.subtitle_ready.connect(self._on_subtitle)
        self.pipeline.metrics_ready.connect(self.metrics.update_values)
        self.pipeline.expression_ready.connect(self.metrics.update_expressions)
        self.pipeline.tracking_quality.connect(self.metrics.update_tracking)
        self.pipeline.status_changed.connect(self._on_status)
        self.pipeline.pipeline_stopped.connect(self.controls.reset)

        # Controls → Pipeline
        self.controls.start_clicked.connect(self._start_pipeline)
        self.controls.stop_clicked.connect(self._stop_pipeline)

    def _setup_timers(self):
        self._metric_timer = QTimer()
        self._metric_timer.timeout.connect(
            lambda: self.metrics.update_values(self.pipeline.profiler.get_latest())
        )

    def _setup_shortcuts(self):
        """Klavye kısayolları — erişilebilirlik."""
        # Space: Başlat/Durdur toggle
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_pipeline)

        # Ctrl+Plus / Ctrl+Minus: Font boyutu
        QShortcut(QKeySequence("Ctrl+="), self, self._font_increase)
        QShortcut(QKeySequence("Ctrl+-"), self, self._font_decrease)

        # Ctrl+S: Altyazı geçmişini kaydet
        QShortcut(QKeySequence("Ctrl+S"), self, self._export_subtitles)

        # F11: Tam ekran toggle
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, self._toggle_fullscreen)

        # Escape: Çıkış
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

        logger.info(
            "Kısayollar: Space=Başlat/Durdur, Ctrl+/-=Font, "
            "Ctrl+S=Export, F11=Tam Ekran, Esc=Çıkış"
        )

    def _toggle_pipeline(self):
        """Space tuşu ile pipeline başlat/durdur."""
        if self.pipeline.running:
            self.controls._on_stop()
        else:
            self.controls._on_start()

    def _font_increase(self):
        """Altyazı font boyutunu büyüt."""
        self._subtitle_font_size = min(self._subtitle_font_size + 2, 32)
        self.subtitle.setStyleSheet(
            self.subtitle.styleSheet().replace(
                f"font-size: {self._subtitle_font_size - 2}px",
                f"font-size: {self._subtitle_font_size}px",
            )
        )
        self.statusBar().showMessage(
            f"Font boyutu: {self._subtitle_font_size}px", 2000
        )

    def _font_decrease(self):
        """Altyazı font boyutunu küçült."""
        self._subtitle_font_size = max(self._subtitle_font_size - 2, 10)
        self.subtitle.setStyleSheet(
            self.subtitle.styleSheet().replace(
                f"font-size: {self._subtitle_font_size + 2}px",
                f"font-size: {self._subtitle_font_size}px",
            )
        )
        self.statusBar().showMessage(
            f"Font boyutu: {self._subtitle_font_size}px", 2000
        )

    def _export_subtitles(self):
        """Altyazı geçmişini .txt dosyasına kaydet."""
        text = self.subtitle.toPlainText()
        if not text.strip():
            self.statusBar().showMessage("Kaydedilecek altyazı yok.", 2000)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Altyazıları Kaydet",
            f"subtitles_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            "Metin Dosyası (*.txt)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.statusBar().showMessage(f"Kaydedildi: {path}", 3000)
            logger.info(f"Altyazılar kaydedildi: {path}")

    def _toggle_fullscreen(self):
        """F11 ile tam ekran toggle."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _start_pipeline(self):
        self.pipeline.start()
        self._metric_timer.start(200)

    def _stop_pipeline(self):
        self._metric_timer.stop()
        self.pipeline.stop()

    def _on_subtitle(self, text: str, conf: float):
        """Subtitle geldiğinde confidence bar'ı güncelle."""
        self.metrics.update_confidence(conf)

    def _on_status(self, text: str):
        self.status_label.setText(text)
        self.controls.set_status(text)

        # Status bar dot rengi
        text_lower = text.lower()
        if "çalışıyor" in text_lower or "bağlı" in text_lower:
            color = Theme.SUCCESS
        elif "mock" in text_lower:
            color = Theme.WARNING
        elif "durdur" in text_lower or "hata" in text_lower:
            color = Theme.ERROR
        else:
            color = Theme.TEXT_MUTED
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 10px;")

    def closeEvent(self, event: QCloseEvent):
        if self.pipeline.running:
            reply = QMessageBox.question(
                self,
                "Çıkış",
                "Pipeline çalışıyor. Durdurup çıkmak istiyor musunuz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        self._metric_timer.stop()
        self.pipeline.stop()
        super().closeEvent(event)
