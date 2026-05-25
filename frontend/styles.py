"""
Blind Eye — Premium Tema Sistemi
Koyu lacivert zemin, cyan/yeşil vurgu, turuncu uyarı.
PyQt6 QSS stillerini merkezi olarak yönetir.
"""


class Theme:
    # ── Renk Paleti ──
    BG_DEEP = "#080c16"
    BG_PRIMARY = "#0a0e1a"
    BG_SECONDARY = "#111827"
    BG_CARD = "#151c2c"
    BG_INPUT = "#1a2236"
    BG_HOVER = "#1e293b"

    BORDER = "#1e293b"
    BORDER_FOCUS = "#00d4aa"

    ACCENT = "#00d4aa"          # Cyan-yeşil (birincil vurgu)
    ACCENT_DIM = "#00a88a"
    ACCENT_GLOW = "rgba(0, 212, 170, 0.15)"
    ACCENT_GRADIENT_START = "#00d4aa"
    ACCENT_GRADIENT_END = "#00b4d8"

    WARNING = "#ff6b35"         # Turuncu uyarı
    ERROR = "#ef4444"           # Kırmızı hata
    SUCCESS = "#22c55e"         # Yeşil başarı
    INFO = "#3b82f6"            # Mavi bilgi

    TEXT_PRIMARY = "#f1f5f9"    # Ana metin
    TEXT_SECONDARY = "#94a3b8"  # İkincil metin
    TEXT_MUTED = "#64748b"      # Soluk metin

    CONFIDENCE_HIGH = "#22c55e"     # >= 0.8
    CONFIDENCE_MED = "#eab308"      # >= 0.5
    CONFIDENCE_LOW = "#ef4444"      # < 0.5

    # ── Duygu-Reaktif Renkler (HUD + Video Overlay) ──
    EMOTION_HAPPY = "#00ffcc"       # Neon cyan-yeşil (Gülümseme)
    EMOTION_ANGRY = "#ff2255"       # Neon kırmızı (Kaş Çatma)
    EMOTION_SURPRISE = "#ffaa00"    # Neon turuncu (Şaşırma)
    EMOTION_NEUTRAL = "#4488ff"     # Soğuk mavi (Nötr)
    EMOTION_MICRO = "#ff00ff"       # Magenta (Mikro-ifade flash)
    EMOTION_DUCHENNE = "#00ff88"    # Parlak yeşil (Duchenne gülümseme)

    # ── Bilişsel Yük Gradyanı ──
    COGNITIVE_LOW = "#22c55e"       # Yeşil (düşük yük — Optimal)
    COGNITIVE_MED = "#eab308"       # Sarı (orta yük — Normal)
    COGNITIVE_HIGH = "#ef4444"      # Kırmızı (yüksek yük — Yorgun/Tehlike)

    # ── Takip Kalitesi ──
    TRACKING_GOOD = "#00d4aa"       # Cyan-yeşil (kaliteli takip)
    TRACKING_POOR = "#ff6b35"       # Turuncu (düşük kalite)

    # ── Tipografi ──
    FONT_FAMILY = "'Segoe UI', 'Inter', 'Roboto', sans-serif"
    FONT_MONO = "'Cascadia Code', 'Consolas', monospace"

    # ── Boyutlar ──
    RADIUS_SM = "6px"
    RADIUS_MD = "10px"
    RADIUS_LG = "14px"

    @classmethod
    def confidence_color(cls, conf: float) -> str:
        if conf >= 0.8:
            return cls.CONFIDENCE_HIGH
        elif conf >= 0.5:
            return cls.CONFIDENCE_MED
        return cls.CONFIDENCE_LOW

    @classmethod
    def emotion_color(cls, dominant: str) -> str:
        """Dominant duyguya göre HUD rengi döndürür."""
        mapping = {
            "Gülümseme": cls.EMOTION_HAPPY,
            "Kaş Çatma": cls.EMOTION_ANGRY,
            "Şaşırma": cls.EMOTION_SURPRISE,
            "Nötr": cls.EMOTION_NEUTRAL,
        }
        return mapping.get(dominant, cls.EMOTION_NEUTRAL)

    @classmethod
    def cognitive_color(cls, load: float) -> str:
        """Bilişsel yük indeksine göre gradyan renk döndürür."""
        if load < 0.3:
            return cls.COGNITIVE_LOW
        elif load < 0.6:
            return cls.COGNITIVE_MED
        return cls.COGNITIVE_HIGH

    @classmethod
    def tracking_color(cls, quality: float) -> str:
        """Takip kalitesine göre renk döndürür."""
        if quality >= 0.6:
            return cls.TRACKING_GOOD
        return cls.TRACKING_POOR


# ═══════════════════════════════════════════════════════════════
#  GLOBAL QSS — Premium koyu tema
# ═══════════════════════════════════════════════════════════════

GLOBAL_QSS = f"""
/* ── Genel ── */
QMainWindow, QWidget {{
    background-color: {Theme.BG_PRIMARY};
    color: {Theme.TEXT_PRIMARY};
    font-family: {Theme.FONT_FAMILY};
    font-size: 13px;
}}

/* ── Başlık çubuğu ── */
QMenuBar {{
    background-color: {Theme.BG_DEEP};
    color: {Theme.TEXT_SECONDARY};
    border-bottom: 1px solid {Theme.BORDER};
    padding: 2px 8px;
}}
QMenuBar::item:selected {{
    background-color: {Theme.BG_HOVER};
    color: {Theme.ACCENT};
}}

/* ── Butonlar ── */
QPushButton {{
    background-color: {Theme.BG_CARD};
    color: {Theme.TEXT_PRIMARY};
    border: 1px solid {Theme.BORDER};
    border-radius: {Theme.RADIUS_MD};
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {Theme.BG_HOVER};
    border-color: {Theme.ACCENT};
    color: {Theme.ACCENT};
}}
QPushButton:pressed {{
    background-color: {Theme.BG_INPUT};
}}
QPushButton:disabled {{
    background-color: {Theme.BG_DEEP};
    color: {Theme.TEXT_MUTED};
    border-color: {Theme.BG_SECONDARY};
}}

/* ── Start butonu (vurgu) ── */
QPushButton#btn_start {{
    background-color: {Theme.ACCENT};
    color: {Theme.BG_DEEP};
    border: none;
    font-weight: 700;
}}
QPushButton#btn_start:hover {{
    background-color: {Theme.ACCENT_DIM};
    color: {Theme.BG_DEEP};
}}
QPushButton#btn_start:disabled {{
    background-color: {Theme.BG_CARD};
    color: {Theme.TEXT_MUTED};
}}

/* ── Stop butonu ── */
QPushButton#btn_stop {{
    background-color: transparent;
    color: {Theme.ERROR};
    border: 1px solid {Theme.ERROR};
}}
QPushButton#btn_stop:hover {{
    background-color: rgba(239, 68, 68, 0.1);
}}

/* ── QLabel ── */
QLabel {{
    color: {Theme.TEXT_PRIMARY};
    background: transparent;
}}

/* ── QTextEdit ── */
QTextEdit {{
    background-color: {Theme.BG_CARD};
    color: {Theme.TEXT_PRIMARY};
    border: 1px solid {Theme.BORDER};
    border-radius: {Theme.RADIUS_MD};
    padding: 8px;
    font-size: 15px;
    selection-background-color: {Theme.ACCENT_GLOW};
}}

/* ── QSpinBox ── */
QSpinBox {{
    background-color: {Theme.BG_INPUT};
    color: {Theme.TEXT_PRIMARY};
    border: 1px solid {Theme.BORDER};
    border-radius: {Theme.RADIUS_SM};
    padding: 6px 8px;
}}
QSpinBox:focus {{
    border-color: {Theme.ACCENT};
}}

/* ── QCheckBox ── */
QCheckBox {{
    color: {Theme.TEXT_SECONDARY};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {Theme.BORDER};
    border-radius: 4px;
    background: {Theme.BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {Theme.ACCENT};
    border-color: {Theme.ACCENT};
}}

/* ── QGroupBox ── */
QGroupBox {{
    background-color: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: {Theme.RADIUS_MD};
    padding: 16px 12px 12px 12px;
    margin-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {Theme.TEXT_SECONDARY};
    font-size: 12px;
}}

/* ── QProgressBar ── */
QProgressBar {{
    background-color: {Theme.BG_DEEP};
    border: 1px solid {Theme.BORDER};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 0px;
}}
QProgressBar::chunk {{
    background-color: {Theme.ACCENT};
    border-radius: 3px;
}}

/* ── QSplitter ── */
QSplitter::handle {{
    background-color: {Theme.BORDER};
    width: 2px;
}}
QSplitter::handle:hover {{
    background-color: {Theme.ACCENT};
}}

/* ── QScrollBar ── */
QScrollBar:vertical {{
    background: {Theme.BG_DEEP};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {Theme.BG_HOVER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Theme.TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── StatusBar ── */
QStatusBar {{
    background-color: {Theme.BG_DEEP};
    color: {Theme.TEXT_MUTED};
    border-top: 1px solid {Theme.BORDER};
    font-size: 11px;
    padding: 2px 8px;
}}

/* ── ToolTip ── */
QToolTip {{
    background-color: {Theme.BG_CARD};
    color: {Theme.TEXT_PRIMARY};
    border: 1px solid {Theme.BORDER};
    border-radius: {Theme.RADIUS_SM};
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Separator ── */
QFrame#separator {{
    background-color: {Theme.BORDER};
    max-height: 1px;
    min-height: 1px;
}}
"""
