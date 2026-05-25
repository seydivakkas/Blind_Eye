from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt
from .styles import Theme


class _MetricCard(QWidget):
    """Tek bir metrik kartı — ikon + değer + birim."""

    def __init__(self, icon: str, label: str, unit: str, color: str):
        super().__init__()
        self.color = color
        self.unit = unit

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Theme.BG_CARD};
                border: 1px solid {Theme.BORDER};
                border-radius: {Theme.RADIUS_MD};
                padding: 0px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        # Üst satır: ikon + etiket
        top = QHBoxLayout()
        top.setSpacing(6)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 14px; color: {color};")

        label_lbl = QLabel(label)
        label_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_MUTED}; font-weight: 500;"
        )

        top.addWidget(icon_lbl)
        top.addWidget(label_lbl)
        top.addStretch()

        # Değer
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {Theme.TEXT_PRIMARY}; "
            f"font-family: {Theme.FONT_MONO};"
        )

        # Birim
        self.unit_label = QLabel(unit)
        self.unit_label.setStyleSheet(
            f"font-size: 10px; color: {Theme.TEXT_MUTED};"
        )

        layout.addLayout(top)
        layout.addWidget(self.value_label)
        layout.addWidget(self.unit_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


def _make_section_card(icon: str, title: str) -> tuple:
    """Yeniden kullanılabilir bölüm kartı oluşturur (konteyner + layout)."""
    container = QWidget()
    container.setStyleSheet(f"""
        QWidget {{
            background-color: {Theme.BG_CARD};
            border: 1px solid {Theme.BORDER};
            border-radius: {Theme.RADIUS_MD};
        }}
    """)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)

    header = QHBoxLayout()
    icon_lbl = QLabel(icon)
    icon_lbl.setStyleSheet("font-size: 14px;")
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(
        f"font-size: 11px; color: {Theme.TEXT_MUTED}; font-weight: 500;"
    )
    header.addWidget(icon_lbl)
    header.addWidget(title_lbl)
    header.addStretch()

    layout.addLayout(header)
    return container, layout, header


def _make_metric_row(label: str, bar_color: str) -> tuple:
    """Yeniden kullanılabilir progress bar satırı oluşturur."""
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setStyleSheet(
        f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; "
        f"min-width: 65px; border: none;"
    )
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(6)
    bar.setStyleSheet(f"""
        QProgressBar {{
            background-color: {Theme.BG_DEEP};
            border: 1px solid {Theme.BORDER};
            border-radius: 3px;
        }}
        QProgressBar::chunk {{
            background-color: {bar_color};
            border-radius: 2px;
        }}
    """)
    val = QLabel("0%")
    val.setStyleSheet(
        f"font-size: 11px; color: {Theme.TEXT_MUTED}; min-width: 32px; "
        f"font-family: {Theme.FONT_MONO}; border: none; text-align: right;"
    )
    row.addWidget(lbl)
    row.addWidget(bar, stretch=1)
    row.addWidget(val)
    return row, bar, val


def _make_separator():
    """İnce ayırıcı çizgi oluşturur."""
    sep = QLabel()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background-color: {Theme.BORDER}; margin: 2px 0;")
    return sep


class MetricsPanel(QWidget):
    """Canlı metrik paneli — Performans + Confidence + Mimik + Kinematik + Bilişsel Yük + Takip."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ═══════════════════════════════════════════════════
        #  1. Performans Kartları (2×2 grid)
        # ═══════════════════════════════════════════════════
        title = QLabel("Performans")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Theme.TEXT_SECONDARY}; "
            f"letter-spacing: 1px; text-transform: uppercase; padding: 0 0 4px 2px;"
        )
        layout.addWidget(title)

        self.latency = _MetricCard("⏱", "Gecikme", "ms", Theme.ACCENT)
        self.fps = _MetricCard("◉", "FPS", "frame/s", Theme.SUCCESS)
        self.cpu = _MetricCard("▪", "CPU", "%", Theme.WARNING)
        self.memory = _MetricCard("▫", "RAM", "MB", "#8b5cf6")

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self.latency)
        row1.addWidget(self.fps)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.cpu)
        row2.addWidget(self.memory)

        layout.addLayout(row1)
        layout.addLayout(row2)

        # ═══════════════════════════════════════════════════
        #  2. Model Güveni (Confidence Bar)
        # ═══════════════════════════════════════════════════
        conf_container, conf_layout, conf_header = _make_section_card("🎯", "Model Güveni")
        self.conf_value = QLabel("--")
        self.conf_value.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {Theme.TEXT_PRIMARY}; "
            f"font-family: {Theme.FONT_MONO};"
        )
        conf_header.addWidget(self.conf_value)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(8)
        conf_layout.addWidget(self.conf_bar)
        layout.addWidget(conf_container)

        # ═══════════════════════════════════════════════════
        #  3. Mimik & Duygu Analizi
        # ═══════════════════════════════════════════════════
        expr_container, expr_layout, expr_header = _make_section_card("🎭", "Mimik Analizi")
        self.expr_value = QLabel("😐 NOTR %100")
        self.expr_value.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Theme.ACCENT}; "
            f"font-family: {Theme.FONT_MONO};"
        )
        expr_header.addWidget(self.expr_value)

        expr_layout.addWidget(_make_separator())

        # Gülümseme
        smile_row, self.smile_bar, self.smile_val = _make_metric_row("Gülümseme", Theme.SUCCESS)
        expr_layout.addLayout(smile_row)

        # Kaş Çatma
        frown_row, self.frown_bar, self.frown_val = _make_metric_row("Kaş Çatma", Theme.ERROR)
        expr_layout.addLayout(frown_row)

        # Şaşırma
        surprise_row, self.surprise_bar, self.surprise_val = _make_metric_row("Şaşırma", Theme.WARNING)
        expr_layout.addLayout(surprise_row)

        layout.addWidget(expr_container)

        # ═══════════════════════════════════════════════════
        #  4. Kinematik Mimik Analizi (YENİ)
        # ═══════════════════════════════════════════════════
        kin_container, kin_layout, kin_header = _make_section_card("⚡", "Kinematik Analiz")

        self.kin_status = QLabel("--")
        self.kin_status.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Theme.ACCENT}; "
            f"font-family: {Theme.FONT_MONO};"
        )
        kin_header.addWidget(self.kin_status)
        kin_layout.addWidget(_make_separator())

        # Mikro-İfade durumu
        micro_row = QHBoxLayout()
        micro_lbl = QLabel("Mikro-İfade")
        micro_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        self.micro_value = QLabel("—")
        self.micro_value.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {Theme.TEXT_MUTED}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )
        micro_row.addWidget(micro_lbl)
        micro_row.addStretch()
        micro_row.addWidget(self.micro_value)
        kin_layout.addLayout(micro_row)

        # Duchenne durumu
        duchenne_row = QHBoxLayout()
        duchenne_lbl = QLabel("Duchenne")
        duchenne_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        self.duchenne_value = QLabel("✗")
        self.duchenne_value.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Theme.TEXT_MUTED}; border: none;"
        )
        duchenne_row.addWidget(duchenne_lbl)
        duchenne_row.addStretch()
        duchenne_row.addWidget(self.duchenne_value)
        kin_layout.addLayout(duchenne_row)

        # Duygu geçişi
        transition_row = QHBoxLayout()
        transition_lbl = QLabel("Geçiş")
        transition_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        self.transition_value = QLabel("—")
        self.transition_value.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_MUTED}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )
        transition_row.addWidget(transition_lbl)
        transition_row.addStretch()
        transition_row.addWidget(self.transition_value)
        kin_layout.addLayout(transition_row)

        layout.addWidget(kin_container)

        # ═══════════════════════════════════════════════════
        #  5. Bilişsel Yük & Yorgunluk (YENİ)
        # ═══════════════════════════════════════════════════
        cog_container, cog_layout, cog_header = _make_section_card("🧠", "Bilişsel Yük")

        self.fatigue_label = QLabel("Optimal")
        self.fatigue_label.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Theme.COGNITIVE_LOW}; "
            f"font-family: {Theme.FONT_MONO};"
        )
        cog_header.addWidget(self.fatigue_label)
        cog_layout.addWidget(_make_separator())

        # Bilişsel yük indeksi barı
        load_row, self.cognitive_bar, self.cognitive_val = _make_metric_row(
            "Yük İndeksi", Theme.COGNITIVE_LOW
        )
        cog_layout.addLayout(load_row)

        # EAR
        ear_row = QHBoxLayout()
        ear_lbl = QLabel("EAR")
        ear_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        ear_lbl.setToolTip("Eye Aspect Ratio — Göz açıklık oranı (Soukupová & Čech, 2016)")
        self.ear_value = QLabel("0.00")
        self.ear_value.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {Theme.TEXT_PRIMARY}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )
        ear_row.addWidget(ear_lbl)
        ear_row.addStretch()
        ear_row.addWidget(self.ear_value)
        cog_layout.addLayout(ear_row)

        # Kırpma/dakika
        blink_row = QHBoxLayout()
        blink_lbl = QLabel("Kırpma/dk")
        blink_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        self.blink_value = QLabel("0")
        self.blink_value.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {Theme.TEXT_PRIMARY}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )
        blink_row.addWidget(blink_lbl)
        blink_row.addStretch()
        blink_row.addWidget(self.blink_value)
        cog_layout.addLayout(blink_row)

        # PERCLOS
        perclos_row, self.perclos_bar, self.perclos_val = _make_metric_row(
            "PERCLOS", Theme.INFO
        )
        cog_layout.addLayout(perclos_row)

        layout.addWidget(cog_container)

        # ═══════════════════════════════════════════════════
        #  6. Takip Kalitesi (YENİ)
        # ═══════════════════════════════════════════════════
        track_container, track_layout, track_header = _make_section_card("🔍", "Takip Kalitesi")

        self.tracking_mode_label = QLabel("—")
        self.tracking_mode_label.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {Theme.ACCENT}; "
            f"font-family: {Theme.FONT_MONO};"
        )
        track_header.addWidget(self.tracking_mode_label)
        track_layout.addWidget(_make_separator())

        # Kalite barı
        quality_row, self.tracking_bar, self.tracking_val = _make_metric_row(
            "KLT Kalitesi", Theme.TRACKING_GOOD
        )
        track_layout.addLayout(quality_row)

        # Mod göstergesi
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Aktif Mod")
        mode_lbl.setStyleSheet(
            f"font-size: 11px; color: {Theme.TEXT_SECONDARY}; min-width: 80px; border: none;"
        )
        self.mode_value = QLabel("Algılama")
        self.mode_value.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {Theme.TRACKING_GOOD}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )
        mode_row.addWidget(mode_lbl)
        mode_row.addStretch()
        mode_row.addWidget(self.mode_value)
        track_layout.addLayout(mode_row)

        layout.addWidget(track_container)

        layout.addStretch()

    # ═══════════════════════════════════════════════════════
    #  Güncelleme Metotları
    # ═══════════════════════════════════════════════════════

    def update_values(self, metrics: dict):
        """Performans metriklerini günceller."""
        self.latency.set_value(f"{metrics.get('latency_ms', 0):.1f}")
        self.fps.set_value(f"{metrics.get('fps', 0):.0f}")
        self.cpu.set_value(f"{metrics.get('cpu_percent', 0):.0f}")
        self.memory.set_value(f"{metrics.get('memory_mb', 0):.0f}")

    def update_confidence(self, conf: float):
        """Confidence bar ve değer güncelleme."""
        percent = int(conf * 100)
        self.conf_bar.setValue(percent)
        self.conf_value.setText(f"{conf:.0%}")

        color = Theme.confidence_color(conf)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.BG_DEEP};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)

    def update_expressions(self, data: dict):
        """Mimik, kinematik ve bilişsel durumu günceller."""
        if not data:
            return

        dominant = data.get("dominant", "Nötr")
        confidence = data.get("confidence", 0.0)
        scores = data.get("scores", {})

        # ── Mimik Analizi ──
        emoji_map = {
            "Gülümseme": "😊 MUTLU",
            "Kaş Çatma": "😠 KIZGIN",
            "Şaşırma": "😮 SASIRMIS",
            "Nötr": "😐 NOTR"
        }

        # Duygu-reaktif renk
        emotion_color = Theme.emotion_color(dominant)
        self.expr_value.setText(f"{emoji_map.get(dominant, '😐 NOTR')} %{int(confidence * 100)}")
        self.expr_value.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {emotion_color}; "
            f"font-family: {Theme.FONT_MONO};"
        )

        # Gülümseme, Kaş Çatma, Şaşırma barları
        s_val = int(scores.get("Gülümseme", 0.0) * 100)
        self.smile_bar.setValue(s_val)
        self.smile_val.setText(f"{s_val}%")

        f_val = int(scores.get("Kaş Çatma", 0.0) * 100)
        self.frown_bar.setValue(f_val)
        self.frown_val.setText(f"{f_val}%")

        su_val = int(scores.get("Şaşırma", 0.0) * 100)
        self.surprise_bar.setValue(su_val)
        self.surprise_val.setText(f"{su_val}%")

        # ── Kinematik Analiz ──
        kinematic = data.get("kinematic")
        if kinematic:
            self._update_kinematic(kinematic, dominant)

        # ── Bilişsel Yük ──
        cognitive = data.get("cognitive")
        if cognitive:
            self._update_cognitive(cognitive)

    def _update_kinematic(self, kin: dict, dominant: str):
        """Kinematik analiz kartını günceller."""
        # Status
        self.kin_status.setText(dominant)
        self.kin_status.setStyleSheet(
            f"font-size: 12px; font-weight: 700; "
            f"color: {Theme.emotion_color(dominant)}; "
            f"font-family: {Theme.FONT_MONO};"
        )

        # Mikro-ifade
        micro = kin.get("micro_expression")
        if micro:
            self.micro_value.setText(micro)
            self.micro_value.setStyleSheet(
                f"font-size: 11px; font-weight: 600; color: {Theme.EMOTION_MICRO}; "
                f"font-family: {Theme.FONT_MONO}; border: none;"
            )
        else:
            self.micro_value.setText("—")
            self.micro_value.setStyleSheet(
                f"font-size: 11px; font-weight: 600; color: {Theme.TEXT_MUTED}; "
                f"font-family: {Theme.FONT_MONO}; border: none;"
            )

        # Duchenne
        is_duchenne = kin.get("is_duchenne", False)
        if is_duchenne:
            self.duchenne_value.setText("✓ Samimi")
            self.duchenne_value.setStyleSheet(
                f"font-size: 12px; font-weight: 700; color: {Theme.EMOTION_DUCHENNE}; border: none;"
            )
        else:
            self.duchenne_value.setText("✗")
            self.duchenne_value.setStyleSheet(
                f"font-size: 12px; font-weight: 700; color: {Theme.TEXT_MUTED}; border: none;"
            )

        # Duygu geçişi
        transition = kin.get("emotion_transition")
        if transition:
            self.transition_value.setText(transition)
            self.transition_value.setStyleSheet(
                f"font-size: 11px; color: {Theme.ACCENT}; "
                f"font-family: {Theme.FONT_MONO}; border: none;"
            )
        else:
            self.transition_value.setText("—")
            self.transition_value.setStyleSheet(
                f"font-size: 11px; color: {Theme.TEXT_MUTED}; "
                f"font-family: {Theme.FONT_MONO}; border: none;"
            )

    def _update_cognitive(self, cog: dict):
        """Bilişsel yük kartını günceller."""
        # Bilişsel yük indeksi
        load = cog.get("cognitive_load", 0.0)
        load_pct = int(load * 100)
        self.cognitive_bar.setValue(load_pct)
        self.cognitive_val.setText(f"{load_pct}%")

        # Bilişsel yük barı renk güncelleme
        cog_color = Theme.cognitive_color(load)
        self.cognitive_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.BG_DEEP};
                border: 1px solid {Theme.BORDER};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {cog_color};
                border-radius: 2px;
            }}
        """)

        # Yorgunluk seviyesi
        fatigue = cog.get("fatigue_level", "Optimal")
        self.fatigue_label.setText(fatigue)
        self.fatigue_label.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {cog_color}; "
            f"font-family: {Theme.FONT_MONO};"
        )

        # EAR
        ear = cog.get("ear", 0.0)
        self.ear_value.setText(f"{ear:.2f}")

        # Kırpma/dakika
        blink_rate = cog.get("blink_rate", 0.0)
        self.blink_value.setText(f"{blink_rate:.0f}")

        # PERCLOS
        perclos = cog.get("perclos", 0.0)
        perclos_pct = int(perclos * 100)
        self.perclos_bar.setValue(perclos_pct)
        self.perclos_val.setText(f"{perclos_pct}%")

    def update_tracking(self, quality: float, mode: str):
        """Takip kalitesi kartını günceller."""
        # Kalite barı
        q_pct = int(quality * 100)
        self.tracking_bar.setValue(q_pct)
        self.tracking_val.setText(f"{q_pct}%")

        track_color = Theme.tracking_color(quality)
        self.tracking_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.BG_DEEP};
                border: 1px solid {Theme.BORDER};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {track_color};
                border-radius: 2px;
            }}
        """)

        # Mod göstergesi
        mode_text = "KLT Takip" if mode == "tracking" else "FaceMesh"
        self.mode_value.setText(mode_text)
        self.mode_value.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {track_color}; "
            f"font-family: {Theme.FONT_MONO}; border: none;"
        )

        # Başlıktaki durum
        self.tracking_mode_label.setText(f"{q_pct}%")
        self.tracking_mode_label.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {track_color}; "
            f"font-family: {Theme.FONT_MONO};"
        )
