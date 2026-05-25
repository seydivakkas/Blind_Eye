"""
backend/expression_detector.py
───────────────────────────────
Geometrik Mimik Tespiti + Zamansal Kinematik Analiz + Bilişsel Yük İzleme.

Üç katmanlı analiz mimarisi:
    1. Statik Geometrik Analiz: Anlık mesafe oranlarından temel duygu skoru
    2. Zamansal Kinematik Analiz: Hız/İvme türevleri ile mikro-ifade ve Duchenne tespiti
    3. Bilişsel Yük İzleme: EAR + PERCLOS ile yorgunluk ve dikkat ölçümü

Akademik referanslar:
    - Ekman & Friesen (1978): FACS
    - Soukupová & Čech (2016): EAR blink detection
    - Yan et al. (2014): Micro-expression analysis
    - Cohn & Schmidt (2004): Duchenne smile timing
"""

import numpy as np
import logging
from typing import Dict, Any

from .kinematic_analyzer import KinematicAnalyzer
from .cognitive_monitor import CognitiveMonitor

logger = logging.getLogger(__name__)


class ExpressionDetector:
    """
    MediaPipe FaceMesh nirengi noktalarını kullanarak 
    çok katmanlı mimik, duygu ve bilişsel durum analizi yapar.
    """

    def __init__(self, fps: float = 30.0):
        # ── Geometrik Analiz İndisleri ──

        # Kararlı mesafe normalizasyonu için gözlerin dış kenar indisleri
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263

        # Kaşlar (Kaş çatma tespiti için iç uçlar)
        self.LEFT_EYEBROW_INNER = 107
        self.RIGHT_EYEBROW_INNER = 336

        # Dudak dış sınırları (Gülümseme ve ağız açıklığı için)
        self.LIP_LEFT_CORNER = 61
        self.LIP_RIGHT_CORNER = 291
        self.LIP_TOP = 0
        self.LIP_BOTTOM = 17

        # Dudak iç sınırları (Dikey açıklık - şaşırma için)
        self.LIP_INNER_TOP = 13
        self.LIP_INNER_BOTTOM = 14

        # Göz dikey açıklıkları (Şaşırma/korku tespiti için)
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        # ── Zamansal Kinematik Analiz Motoru ──
        self.kinematic = KinematicAnalyzer(buffer_size=30, fps=fps)

        # ── Bilişsel Yük İzleyici ──
        self.cognitive = CognitiveMonitor(fps=fps)

    def _dist(self, p1: Any, p2: Any) -> float:
        """İki nokta arasındaki Öklid uzaklığı."""
        return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))

    def detect(self, landmarks) -> Dict[str, Any]:
        """
        Nirengi noktalarından çok katmanlı analiz yapar.

        Katman 1: Geometrik mimik skorları (statik)
        Katman 2: Kinematik türev analizi (zamansal)
        Katman 3: Bilişsel yük ve yorgunluk (EAR/PERCLOS)
        """
        if not landmarks or len(landmarks) < 468:
            return {
                "dominant": "Nötr",
                "confidence": 1.0,
                "scores": {"Gülümseme": 0.0, "Kaş Çatma": 0.0, "Şaşırma": 0.0},
                "kinematic": {
                    "velocities": {},
                    "accelerations": {},
                    "micro_expression": None,
                    "is_duchenne": False,
                    "emotion_transition": None,
                },
                "cognitive": {
                    "ear": 0.0,
                    "blink_rate": 0.0,
                    "perclos": 0.0,
                    "cognitive_load": 0.0,
                    "fatigue_level": "Optimal",
                },
            }

        # ═══════════════════════════════════════════════════
        #  KATMAN 1: Geometrik Mimik Skorları (Statik)
        # ═══════════════════════════════════════════════════

        # 1. Normalizasyon skalası (Gözlerin dış kenarları arasındaki mesafe)
        base_scale = self._dist(landmarks[self.LEFT_EYE_OUTER], landmarks[self.RIGHT_EYE_OUTER])
        if base_scale < 1e-5:
            base_scale = 1.0

        # 2. Gülümseme (Smile) Skoru
        # Ağız genişliği
        mouth_w = self._dist(landmarks[self.LIP_LEFT_CORNER], landmarks[self.LIP_RIGHT_CORNER])
        norm_mouth_w = mouth_w / base_scale

        # Ağız kenarlarının yüksekliği (Dudak merkezine göre ne kadar yukarıda)
        lip_center_y = (landmarks[self.LIP_TOP].y + landmarks[self.LIP_BOTTOM].y) / 2.0
        avg_corner_y = (landmarks[self.LIP_LEFT_CORNER].y + landmarks[self.LIP_RIGHT_CORNER].y) / 2.0
        raise_y = lip_center_y - avg_corner_y
        norm_raise_y = raise_y / base_scale

        # Gülümseme formülü
        # Nötr ağız genişliği oranı genelde 0.65-0.70 arasındadır, gülümseme ile 0.85+ seviyesine çıkar.
        # Nötr ağız köşeleri yüksekliği 0.0 civarıdır, gülümseme ile yukarı kıvrılır (> 0.05).
        smile_score = (norm_mouth_w - 0.68) * 3.0 + (norm_raise_y - 0.01) * 6.0
        smile_score = float(np.clip(smile_score, 0.0, 1.0))

        # 3. Kaş Çatma (Frown/Anger) Skoru
        # İç kaşlar arası mesafe (ne kadar yakınsa o kadar çatık)
        eyebrow_dist = self._dist(landmarks[self.LEFT_EYEBROW_INNER], landmarks[self.RIGHT_EYEBROW_INNER])
        norm_eyebrow_dist = eyebrow_dist / base_scale

        # Çatık kaş formülü (Nötr durumda oran ~0.48, çatıldığında ~0.38 veya daha az)
        frown_score = (0.47 - norm_eyebrow_dist) * 8.0
        frown_score = float(np.clip(frown_score, 0.0, 1.0))

        # 4. Şaşırma (Surprise) Skoru
        # Ağız dikey açıklığı
        inner_mouth_h = self._dist(landmarks[self.LIP_INNER_TOP], landmarks[self.LIP_INNER_BOTTOM])
        norm_inner_mouth_h = inner_mouth_h / base_scale

        # Gözlerin açıklık miktarı
        left_eye_h = self._dist(landmarks[self.LEFT_EYE_TOP], landmarks[self.LEFT_EYE_BOTTOM])
        right_eye_h = self._dist(landmarks[self.RIGHT_EYE_TOP], landmarks[self.RIGHT_EYE_BOTTOM])
        norm_eye_h = ((left_eye_h + right_eye_h) / 2.0) / base_scale

        # Şaşırma formülü (Gözlerin genişlemesi ve ağzın açılması kombinasyonu)
        # Nötr göz açıklık oranı ~0.14, şaşırmada ~0.20+.
        # Nötr iç dudak dikey aralığı ~0.01, şaşırmada ~0.15+.
        surprise_score = (norm_inner_mouth_h - 0.05) * 3.0 + (norm_eye_h - 0.15) * 4.0
        surprise_score = float(np.clip(surprise_score, 0.0, 1.0))

        # 5. Baskın Duygu/Mimik Belirleme
        scores = {
            "Gülümseme": smile_score,
            "Kaş Çatma": frown_score,
            "Şaşırma": surprise_score
        }

        dominant = "Nötr"
        max_score = 0.0
        threshold = 0.35  # Eşik değerinin altı nötr kabul edilir

        for name, val in scores.items():
            if val > max_score:
                max_score = val
                dominant = name

        if max_score < threshold:
            dominant = "Nötr"
            confidence = 1.0 - max(scores.values())
        else:
            confidence = max_score

        # ═══════════════════════════════════════════════════
        #  KATMAN 2: Zamansal Kinematik Analiz
        # ═══════════════════════════════════════════════════

        # Geometrik feature'ları kinematik analizöre besle
        norm_left_eye_h = left_eye_h / base_scale
        norm_right_eye_h = right_eye_h / base_scale

        kinematic_features = {
            "mouth_width": norm_mouth_w,
            "mouth_height": norm_inner_mouth_h,
            "eyebrow_dist": norm_eyebrow_dist,
            "left_eye_height": norm_left_eye_h,
            "right_eye_height": norm_right_eye_h,
            "lip_corner_raise": norm_raise_y,
        }

        kin_state = self.kinematic.update(kinematic_features, dominant=dominant)

        # ═══════════════════════════════════════════════════
        #  KATMAN 3: Bilişsel Yük ve Yorgunluk
        # ═══════════════════════════════════════════════════

        cog_state = self.cognitive.update(landmarks)

        # ═══════════════════════════════════════════════════
        #  Birleşik Çıktı
        # ═══════════════════════════════════════════════════

        return {
            "dominant": dominant,
            "confidence": float(confidence),
            "scores": scores,
            "kinematic": {
                "velocities": kin_state.velocities,
                "accelerations": kin_state.accelerations,
                "micro_expression": kin_state.micro_expression,
                "is_duchenne": kin_state.is_duchenne,
                "emotion_transition": kin_state.emotion_transition,
            },
            "cognitive": {
                "ear": cog_state.ear,
                "is_blinking": cog_state.is_blinking,
                "blink_rate": cog_state.blink_rate,
                "avg_blink_duration_ms": cog_state.avg_blink_duration_ms,
                "perclos": cog_state.perclos,
                "cognitive_load": cog_state.cognitive_load,
                "fatigue_level": cog_state.fatigue_level,
            },
        }
