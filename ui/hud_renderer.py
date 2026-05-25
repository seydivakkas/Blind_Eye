"""
ui/hud_renderer.py
───────────────────
Pi 3 Model B+ Fütüristik Vektörel HUD Render Motoru.

Tasarım Prensipleri:
    - ASLA cv2.addWeighted / alpha blending kullanma (CPU boğar)
    - Sadece cv2.line, cv2.rectangle, cv2.putText, cv2.polylines (vektörel)
    - Duygu-reaktif renk sistemi (dominant duyguya göre HUD rengi değişir)
    - CPU sıcaklığı 2 saniyede bir okunur (File I/O throttle)
    - Foveated rendering: sadece ROI çevresine yüksek detay çizimi

Akademik referanslar:
    - Foveated Rendering: Guenter et al. (2012) — SIGGRAPH
    - Termal Throttling: ARM Cortex-A53 TDP yönetimi
"""

import os
import time
import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Duygu → BGR Renk Eşlemesi
# ═══════════════════════════════════════════════════════════════

EMOTION_BGR = {
    "Gülümseme": (204, 255, 0),      # Neon cyan-yeşil
    "Kaş Çatma": (85, 34, 255),       # Neon kırmızı
    "Şaşırma": (0, 170, 255),         # Neon turuncu
    "Nötr": (255, 200, 100),          # Soğuk mavi
}

FATIGUE_BGR = {
    "Optimal": (0, 197, 34),          # Yeşil
    "Normal": (0, 179, 234),          # Sarı/Cyan
    "Yorgun": (0, 100, 255),          # Turuncu
    "Tehlike": (68, 68, 239),         # Kırmızı
}

TRACKING_BGR = {
    True: (170, 212, 0),              # Takip modu — cyan
    False: (0, 170, 255),             # Algılama modu — turuncu
}


class HUDRenderer:
    """Pi 3 Model B+ Fütüristik Vektörel HUD Render Motoru.

    Sıfır alpha blending ile düşük CPU yükünde cyberpunk arayüz çizer.
    CPU sıcaklığını termal dosyadan 2 saniyede bir okur.

    Args:
        panel_width: Sağ yan panelin genişliği (piksel)
        temp_throttle_sec: CPU sıcaklık okuma aralığı (saniye)
    """

    def __init__(self, panel_width: int = 180, temp_throttle_sec: float = 2.0):
        self.panel_width = panel_width
        self.temp_throttle_sec = temp_throttle_sec

        # CPU sıcaklık durumu
        self._last_temp: float = 0.0
        self._last_temp_time: float = 0.0
        self._temp_path = "/sys/class/thermal/thermal_zone0/temp"
        self._temp_available = os.path.exists(self._temp_path)

    # ═══════════════════════════════════════════════════════
    #  CPU Sıcaklık Okuma (Throttled File I/O)
    # ═══════════════════════════════════════════════════════

    def get_cpu_temp(self) -> float:
        """CPU sıcaklığını °C olarak döndürür.

        File I/O darboğazını önlemek için temp_throttle_sec
        saniyede bir okuma yapar, aradaki çağrılarda cache döner.
        """
        current_time = time.time()

        if current_time - self._last_temp_time < self.temp_throttle_sec:
            return self._last_temp

        if not self._temp_available:
            return 0.0

        try:
            with open(self._temp_path, "r") as f:
                raw = f.read().strip()
                self._last_temp = int(raw) / 1000.0
                self._last_temp_time = current_time
        except (IOError, ValueError):
            pass

        return self._last_temp

    # ═══════════════════════════════════════════════════════
    #  Ana HUD Çizim Fonksiyonu
    # ═══════════════════════════════════════════════════════

    def render(
        self,
        frame: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        lip_landmarks: Optional[List[Tuple[int, int]]] = None,
        all_landmarks_px: Optional[List[Tuple[int, int]]] = None,
        expressions: Optional[Dict] = None,
        tracking_mode: str = "detection",
        tracking_quality: float = 0.0,
        fps: int = 0,
        inference_latency: float = 0.0,
        prediction_text: str = "",
        prediction_conf: float = 0.0,
        mimic_mode: bool = True,
    ) -> np.ndarray:
        """Ana HUD render fonksiyonu — tüm overlay'ları frame üzerine çizer.

        Args:
            frame: Çizim yapılacak BGR frame (in-place değiştirilir)
            roi_bbox: (x1, y1, x2, y2) dudak ROI bounding box
            lip_landmarks: Dudak noktaları [(x, y), ...] piksel
            all_landmarks_px: Tüm yüz noktaları [(x, y), ...] piksel (opsiyonel)
            expressions: ExpressionDetector.detect() çıktısı
            tracking_mode: "detection" veya "tracking"
            tracking_quality: KLT takip kalitesi [0.0 - 1.0]
            fps: Anlık FPS değeri
            inference_latency: ONNX çıkarım süresi (ms)
            prediction_text: Dudak okuma tahmini
            prediction_conf: Tahmin güveni [0.0 - 1.0]
            mimic_mode: Mimik analiz paneli açık mı

        Returns:
            Çizim yapılmış frame (aynı referans)
        """
        h, w = frame.shape[:2]

        # Duygu-reaktif ana renk
        dominant = "Nötr"
        if expressions:
            dominant = expressions.get("dominant", "Nötr")
        hud_color = EMOTION_BGR.get(dominant, EMOTION_BGR["Nötr"])

        # ── 1. Foveated ROI Corner Markers ──
        if roi_bbox is not None:
            self._draw_corner_markers(frame, roi_bbox, hud_color)

        # ── 2. Yüz Mesh Overlay (vektörel) ──
        if all_landmarks_px:
            self._draw_face_mesh(frame, all_landmarks_px, hud_color, expressions)

        # ── 3. Dudak Polyline ──
        if lip_landmarks and len(lip_landmarks) > 2:
            pts = np.array(lip_landmarks, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=hud_color,
                          thickness=1, lineType=cv2.LINE_AA)

        # ── 4. Duygu Etiketi (ROI üstü) ──
        if roi_bbox is not None and expressions:
            self._draw_emotion_label(frame, roi_bbox, expressions, hud_color)

        # ── 5. Bilişsel Yük HUD (ROI altı) ──
        if roi_bbox is not None and expressions:
            self._draw_cognitive_label(frame, roi_bbox, expressions)

        # ── 6. Sol Üst — Sistem Durumu ──
        self._draw_system_status(frame, fps, inference_latency,
                                 tracking_mode, tracking_quality)

        # ── 7. Sağ Panel — Mimik Analiz Dashboardu ──
        if mimic_mode:
            self._draw_mimic_panel(frame, expressions)

        # ── 8. Alt Panel — Tahmin Çıktısı ──
        self._draw_prediction_bar(frame, prediction_text, prediction_conf,
                                  dominant, mimic_mode)

        return frame

    # ═══════════════════════════════════════════════════════
    #  Yardımcı Çizim Fonksiyonları
    # ═══════════════════════════════════════════════════════

    def _draw_corner_markers(self, frame, bbox, color, thickness=2):
        """Foveated köşe işaretleri — tam dikdörtgen yerine."""
        x1, y1, x2, y2 = bbox
        corner_len = max(12, (x2 - x1) // 5)

        # 4 köşe × 2 çizgi = 8 çizgi
        for (sx, sy), (dx, dy) in [
            ((x1, y1), (1, 0)), ((x1, y1), (0, 1)),   # Sol üst
            ((x2, y1), (-1, 0)), ((x2, y1), (0, 1)),   # Sağ üst
            ((x1, y2), (1, 0)), ((x1, y2), (0, -1)),   # Sol alt
            ((x2, y2), (-1, 0)), ((x2, y2), (0, -1)),   # Sağ alt
        ]:
            cv2.line(frame, (sx, sy),
                     (sx + dx * corner_len, sy + dy * corner_len),
                     color, thickness)

    def _draw_face_mesh(self, frame, landmarks_px, color, expressions):
        """Vektörel yüz mesh çizimi — kaş, göz, dudak."""
        # Kaş çizgileri — kaş çatma skoru ile renk değişimi
        frown_score = 0.0
        if expressions and "scores" in expressions:
            frown_score = expressions["scores"].get("Kaş Çatma", 0.0)

        brow_color = (0, 0, int(100 + 155 * frown_score)) if frown_score > 0.3 else (100, 100, 200)

        # Sol kaş
        brow_l_idx = [70, 63, 105, 66, 107]
        if all(i < len(landmarks_px) for i in brow_l_idx):
            pts = np.array([landmarks_px[i] for i in brow_l_idx], np.int32)
            cv2.polylines(frame, [pts], False, brow_color, 1, cv2.LINE_AA)

        # Sağ kaş
        brow_r_idx = [300, 293, 334, 296, 336]
        if all(i < len(landmarks_px) for i in brow_r_idx):
            pts = np.array([landmarks_px[i] for i in brow_r_idx], np.int32)
            cv2.polylines(frame, [pts], False, brow_color, 1, cv2.LINE_AA)

        # Göz noktaları — bilişsel yük rengine göre
        cog_color = (0, 255, 0)  # Varsayılan yeşil
        if expressions:
            cognitive = expressions.get("cognitive", {})
            fatigue = cognitive.get("fatigue_level", "Optimal")
            cog_color = FATIGUE_BGR.get(fatigue, (0, 255, 0))

        eye_indices = [33, 159, 133, 145, 263, 386, 362, 374]
        for idx in eye_indices:
            if idx < len(landmarks_px):
                cv2.circle(frame, landmarks_px[idx], 2, cog_color, -1)

    def _draw_emotion_label(self, frame, bbox, expressions, hud_color):
        """ROI üstüne duygu + kinematik etiket çizer."""
        x1, y1, x2, y2 = bbox
        dominant = expressions.get("dominant", "Nötr")
        conf = expressions.get("confidence", 0.0)

        labels_map = {
            "Gülümseme": "MUTLU", "Kaş Çatma": "KIZGIN",
            "Şaşırma": "SASIRMIS", "Nötr": "NOTR"
        }
        lbl_text = f"{labels_map.get(dominant, 'NOTR')} %{int(conf * 100)}"

        # Mikro-ifade flash
        kinematic = expressions.get("kinematic", {})
        micro = kinematic.get("micro_expression")
        active_color = hud_color
        if micro:
            lbl_text += f" [{micro}]"
            active_color = (255, 0, 255)  # Magenta

        # Duchenne göstergesi
        if kinematic.get("is_duchenne", False):
            lbl_text += " [D]"

        label_y = max(y1 - 10, 25)
        (w_text, h_text), _ = cv2.getTextSize(
            lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Koyu arka plan kutusu + çerçeve (vektörel, alpha yok)
        cv2.rectangle(frame,
                      (x1, label_y - h_text - 6),
                      (x1 + w_text + 12, label_y + 4),
                      (18, 18, 18), -1)
        cv2.rectangle(frame,
                      (x1, label_y - h_text - 6),
                      (x1 + w_text + 12, label_y + 4),
                      active_color, 1)
        cv2.putText(frame, lbl_text, (x1 + 6, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1, cv2.LINE_AA)

    def _draw_cognitive_label(self, frame, bbox, expressions):
        """ROI altına bilişsel yük / EAR göstergesi çizer."""
        cognitive = expressions.get("cognitive", {})
        fatigue = cognitive.get("fatigue_level", "")
        ear = cognitive.get("ear", 0.0)
        blink_rate = cognitive.get("blink_rate", 0.0)

        if not fatigue:
            return

        x1, _, _, y2 = bbox
        cog_color = FATIGUE_BGR.get(fatigue, (200, 200, 200))
        cog_text = f"EAR:{ear:.2f} Kirp:{blink_rate:.0f}/dk [{fatigue}]"
        cv2.putText(frame, cog_text, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cog_color, 1, cv2.LINE_AA)

    def _draw_system_status(self, frame, fps, latency,
                            tracking_mode, tracking_quality):
        """Sol üst köşe — sistem durumu bilgisi."""
        h, w = frame.shape[:2]
        y_offset = 25
        line_h = 20

        # Sistem kimliği
        cv2.putText(frame, "PI 3 B+ - 4 Cores 1.4GHz", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        y_offset += line_h

        # FPS
        fps_color = (0, 255, 0) if fps >= 20 else (0, 170, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
        y_offset += line_h

        # Inference latency
        cv2.putText(frame, f"Cikarim: {latency:.1f} ms", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += line_h

        # Takip modu + kalite
        is_tracking = tracking_mode == "tracking"
        mode_text = "KLT Takip" if is_tracking else "FaceMesh"
        mode_color = TRACKING_BGR.get(is_tracking, (200, 200, 200))
        cv2.putText(frame, f"Mod: {mode_text} ({int(tracking_quality * 100)}%)",
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
        y_offset += line_h

        # CPU sıcaklığı (throttled)
        temp = self.get_cpu_temp()
        if temp > 0:
            temp_color = (0, 255, 0) if temp < 60 else (0, 170, 255) if temp < 70 else (0, 0, 255)
            cv2.putText(frame, f"CPU.TEMP: {temp:.0f}C", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, temp_color, 1, cv2.LINE_AA)
            y_offset += line_h

        # KVKK
        cv2.putText(frame, "KVKK: RAM-Only", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_mimic_panel(self, frame, expressions):
        """Sağ yan panel — mimik analiz dashboard (vektörel, alpha-free)."""
        h, w = frame.shape[:2]
        px = w - self.panel_width  # Panel sol kenarı

        # Panel arka planı — opak koyu dikdörtgen (alpha blending YOK)
        cv2.rectangle(frame, (px, 0), (w, h), (18, 18, 18), -1)
        # Sınır çizgisi
        cv2.line(frame, (px, 0), (px, h), (50, 50, 50), 1)

        # Panel başlığı
        cv2.putText(frame, "MIMIK ANALIZI", (px + 12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (px + 12, 42), (w - 12, 42), (80, 80, 80), 1)

        if not expressions:
            cv2.putText(frame, "YUZ BEKLENIYOR", (px + 12, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            return

        dominant = expressions.get("dominant", "Nötr")
        conf = expressions.get("confidence", 0.0)
        scores = expressions.get("scores", {})
        kinematic = expressions.get("kinematic", {})
        cognitive = expressions.get("cognitive", {})

        # Dominant duygu
        labels_map = {
            "Gülümseme": "MUTLU", "Kaş Çatma": "KIZGIN",
            "Şaşırma": "SASIRMIS", "Nötr": "NOTR"
        }
        dom_color = EMOTION_BGR.get(dominant, (200, 200, 200))
        cv2.putText(frame, labels_map.get(dominant, "NOTR"), (px + 12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dom_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Guven: %{int(conf * 100)}", (px + 12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

        # Seviye barları
        bar_x = px + 12
        bar_w = self.panel_width - 24
        bar_h = 8

        y = 120
        for name, key, color in [
            ("Gulumseme", "Gülümseme", (76, 175, 80)),
            ("Kas Catma", "Kaş Çatma", (244, 67, 54)),
            ("Sasirma", "Şaşırma", (255, 152, 0)),
        ]:
            val = scores.get(key, 0.0)
            self._draw_bar(frame, name, val, bar_x, y, bar_w, bar_h, color)
            y += 40

        # ── Kinematik bölümü ──
        cv2.line(frame, (px + 12, y), (w - 12, y), (60, 60, 60), 1)
        y += 18
        cv2.putText(frame, "KINEMATIK", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18

        # Mikro-ifade
        micro = kinematic.get("micro_expression")
        micro_text = micro if micro else "--"
        micro_color = (255, 0, 255) if micro else (120, 120, 120)
        cv2.putText(frame, f"Mikro: {micro_text}", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, micro_color, 1, cv2.LINE_AA)
        y += 18

        # Duchenne
        is_duchenne = kinematic.get("is_duchenne", False)
        d_text = "Samimi" if is_duchenne else "--"
        d_color = (0, 255, 136) if is_duchenne else (120, 120, 120)
        cv2.putText(frame, f"Duchenne: {d_text}", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, d_color, 1, cv2.LINE_AA)
        y += 18

        # Duygu geçişi
        transition = kinematic.get("emotion_transition")
        if transition:
            cv2.putText(frame, f"Gecis: {transition}", (px + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
        y += 22

        # ── Bilişsel yük bölümü ──
        cv2.line(frame, (px + 12, y), (w - 12, y), (60, 60, 60), 1)
        y += 18
        cv2.putText(frame, "BILISSEL YUK", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18

        # EAR
        ear = cognitive.get("ear", 0.0)
        cv2.putText(frame, f"EAR: {ear:.2f}", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18

        # Kırpma/dakika
        blink_rate = cognitive.get("blink_rate", 0.0)
        cv2.putText(frame, f"Kirpma: {blink_rate:.0f}/dk", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18

        # PERCLOS
        perclos = cognitive.get("perclos", 0.0)
        cv2.putText(frame, f"PERCLOS: {perclos:.0%}", (px + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20

        # Bilişsel yük barı
        load = cognitive.get("cognitive_load", 0.0)
        fatigue = cognitive.get("fatigue_level", "Optimal")
        fatigue_color = FATIGUE_BGR.get(fatigue, (200, 200, 200))
        self._draw_bar(frame, f"Yuk [{fatigue}]", load, bar_x, y, bar_w, bar_h, fatigue_color)

    def _draw_prediction_bar(self, frame, text, conf, dominant, mimic_mode):
        """Alt panel — tahmin çıktısı."""
        h, w = frame.shape[:2]
        panel_w = w - self.panel_width if mimic_mode else w

        # Koyu alt bar (vektörel)
        cv2.rectangle(frame, (0, h - 55), (panel_w, h), (12, 12, 12), -1)
        cv2.line(frame, (0, h - 55), (panel_w, h - 55), (50, 50, 50), 1)

        # Tahmin metni
        dom_tag = f" | [{dominant}]"
        if text:
            msg = f"TURKCE: '{text}' ({conf * 100:.1f}%){dom_tag}"
            msg_color = (0, 255, 0)
        else:
            msg = f"Konusma bekleniyor...{dom_tag}"
            msg_color = (120, 120, 120)

        cv2.putText(frame, msg, (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, msg_color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_bar(frame, label, value, x, y, w, h, color):
        """Vektörel seviye barı çizer (alpha blending YOK)."""
        # Arka plan
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1)
        # Doluluk
        fill_w = int(w * max(0.0, min(value, 1.0)))
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
        # Etiket
        cv2.putText(frame, f"{label}: %{int(value * 100)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
