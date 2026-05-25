import os
import sys
import logging
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont

from frontend.styles import Theme, GLOBAL_QSS
from frontend.studio_panels import DataStudioTab, TrainingStudioTab, PiGlassesStudioTab, AnalyticsStudioTab

# Console logging ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("studio_main")


class BlindEyeStudioWindow(QtWidgets.QMainWindow):
    """Masaüstü Veri Toplama ve Model Eğitim Stüdyosu — TÜBİTAK 2209-A"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Blind Eye Studio — Veri Toplama & Model Eğitim İstasyonu")
        self.resize(1100, 750)
        self.setMinimumSize(950, 600)
        self.setStyleSheet(GLOBAL_QSS)

        self._setup_ui()

    def _setup_ui(self):
        # Ana widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(10)

        # ── 1. ÜST BAR: Başlık ve Sürüm ──
        header = QtWidgets.QHBoxLayout()
        
        logo = QtWidgets.QLabel("◉ Blind Eye Studio")
        logo.setStyleSheet(
            f"font-size: 20px; font-weight: 800; color: {Theme.ACCENT}; "
            f"letter-spacing: -0.5px;"
        )
        
        version = QtWidgets.QLabel("Akademik Geliştirici & Eğitim Konsolu v1.0")
        version.setStyleSheet(f"font-size: 11px; color: {Theme.TEXT_MUTED}; font-weight: bold;")
        
        header.addWidget(logo)
        header.addWidget(version)
        header.addStretch()
        
        kvkk_badge = QtWidgets.QLabel("🔒 KVKK Uyumlu · RAM-Only Çıkarım")
        kvkk_badge.setStyleSheet(
            f"color: {Theme.ACCENT}; background-color: {Theme.ACCENT_GLOW}; "
            f"border: 1px solid {Theme.ACCENT}; border-radius: 4px; padding: 4px 8px; font-size: 10px; font-weight: bold;"
        )
        header.addWidget(kvkk_badge)
        main_layout.addLayout(header)

        # ── 2. SEKMELİ YAPI: TabWidget ──
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {Theme.BORDER}; border-radius: {Theme.RADIUS_LG}; "
            f"background-color: {Theme.BG_SECONDARY}; top: -1px; }} "
            f"QTabBar::tab {{ background-color: {Theme.BG_PRIMARY}; border: 1px solid {Theme.BORDER}; "
            f"border-bottom-color: transparent; border-top-left-radius: 6px; border-top-right-radius: 6px; "
            f"padding: 8px 20px; font-weight: 600; color: {Theme.TEXT_SECONDARY}; }} "
            f"QTabBar::tab:selected, QTabBar::tab:hover {{ background-color: {Theme.BG_SECONDARY}; "
            f"color: {Theme.ACCENT}; border-bottom-color: {Theme.BG_SECONDARY}; }} "
            f"QTabBar::tab:selected {{ border-top: 2px solid {Theme.ACCENT}; }}"
        )

        # Tab 1: Veri Toplama
        self.data_tab = DataStudioTab()
        self.data_tab.status_msg.connect(self._show_status_message)
        self.tabs.addTab(self.data_tab, "Veri Toplama Stüdyosu")

        # Tab 2: Model Eğitimi
        self.training_tab = TrainingStudioTab()
        self.training_tab.status_msg.connect(self._show_status_message)
        self.tabs.addTab(self.training_tab, "Model Eğitim Konsolu")

        # Tab 3: Giyilebilir Gözlük İstasyonu
        self.glasses_tab = PiGlassesStudioTab()
        self.glasses_tab.status_msg.connect(self._show_status_message)
        self.tabs.addTab(self.glasses_tab, "Giyilebilir Gözlük İstasyonu")

        # Tab 4: Yapay Zeka Analitik & Sıkıştırma
        self.analytics_tab = AnalyticsStudioTab()
        self.analytics_tab.status_msg.connect(self._show_status_message)
        self.tabs.addTab(self.analytics_tab, "Yapay Zeka Analitik & Sıkıştırma")

        main_layout.addWidget(self.tabs)

        # ── 3. STATUS BAR ──
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {Theme.BG_DEEP}; color: {Theme.TEXT_MUTED}; font-size: 11px;")
        self.setStatusBar(self.status_bar)
        
        self.status_dot = QtWidgets.QLabel("●")
        self.status_dot.setStyleSheet(f"color: {Theme.SUCCESS}; font-size: 10px; margin-right: 5px;")
        self.status_bar.addWidget(self.status_dot)

        self.status_text = QtWidgets.QLabel("Stüdyo hazır. Veri kaydı veya eğitimi başlatabilirsiniz.")
        self.status_text.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.status_bar.addWidget(self.status_text)

        # Kısayollar ve İpuçları
        shortcut_tips = QtWidgets.QLabel("İpucu: Model eğitimi arka planda asenkron olarak güvenle koşturulur.")
        shortcut_tips.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 10px;")
        self.status_bar.addPermanentWidget(shortcut_tips)

    def _show_status_message(self, message):
        """Alt durum çubuğunu günceller."""
        self.status_text.setText(message)
        
        # Sinyal tipine göre status dot rengini ayarla
        msg_lower = message.lower()
        if "hata" in msg_lower or "başarısız" in msg_lower or "❌" in msg_lower:
            self.status_dot.setStyleSheet(f"color: {Theme.ERROR}; font-size: 10px;")
        elif "kayıt" in msg_lower or "hazırlanın" in msg_lower or "..." in msg_lower:
            self.status_dot.setStyleSheet(f"color: {Theme.WARNING}; font-size: 10px;")
        elif "başarıyla" in msg_lower or "tamamlandı" in msg_lower or "✅" in msg_lower:
            self.status_dot.setStyleSheet(f"color: {Theme.SUCCESS}; font-size: 10px;")
        else:
            self.status_dot.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 10px;")

    def closeEvent(self, event):
        """Kapatma esnasında asenkron kamera thread'ini ve eğitim işçisini durdurur."""
        # Tab 1 Kamerası
        if hasattr(self, 'data_tab') and self.data_tab.camera_thread.isRunning():
            self.data_tab.camera_thread.stop()

        # Tab 2 Eğitimi
        if hasattr(self, 'training_tab') and self.training_tab.is_training:
            reply = QtWidgets.QMessageBox.question(
                self, "Eğitim Devam Ediyor",
                "Model eğitimi şu an arka planda çalışıyor. Stüdyoyu kapatmak eğitimi durduracaktır. Çıkmak istiyor musunuz?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                event.ignore()
                return
            else:
                self.training_tab.worker.stop()

        # Tab 3 Gözlük Telemetrisi
        if hasattr(self, 'glasses_tab') and hasattr(self.glasses_tab, 'telemetry_worker') and self.glasses_tab.telemetry_worker.isRunning():
            self.glasses_tab.telemetry_worker.stop()

        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Unicode konsol ayarı
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Windows'ta Görev Çubuğunda Ayrı İkon Olarak Gösterilmesi İçin AppID Ayarı
    if sys.platform == "win32":
        import ctypes
        myappid = 'tubitak2209.blindeye.studio.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    window = BlindEyeStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
