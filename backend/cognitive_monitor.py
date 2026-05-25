"""
backend/cognitive_monitor.py
─────────────────────────────
EAR (Eye Aspect Ratio) + PERCLOS Tabanlı Bilişsel Durum İzleyici.

İşitme engelli bireylerin dudak okuma sırasındaki bilişsel yükünü ve
yorgunluğunu gerçek zamanlı olarak ölçer.

Metrikler:
    1. EAR (Eye Aspect Ratio): Göz açıklık oranı
    2. PERCLOS: Son 60 saniyede gözlerin kapalı olduğu yüzde
    3. Kırpma frekansı: Kırpma/dakika
    4. Kırpma süresi: Ortalama kapanma süresi (ms)
    5. Bilişsel yük indeksi: Weighted composite [0.0 - 1.0]

Akademik referanslar:
    - Soukupová & Čech (2016): "Real-Time Eye Blink Detection
      using Facial Landmarks"
    - Dinges & Grace (1998): PERCLOS — Percentage of Eye Closure
    - Wierwille et al. (1994): "Research on Vehicle-Based Driver
      Status/Performance Monitoring"
"""

import numpy as np
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Bilişsel Durum Veri Yapısı
# ═══════════════════════════════════════════════════════════════

@dataclass
class CognitiveState:
    """Anlık bilişsel durum ölçümleri."""
    ear: float = 0.0                       # Anlık EAR değeri
    is_blinking: bool = False              # Göz şu an kapalı mı
    blink_count: int = 0                   # Toplam kırpma sayısı
    blink_rate: float = 0.0                # Kırpma/dakika
    avg_blink_duration_ms: float = 0.0     # Ort. kırpma süresi
    perclos: float = 0.0                   # PERCLOS yüzdesi [0.0 - 1.0]
    cognitive_load: float = 0.0            # Bilişsel yük indeksi [0.0 - 1.0]
    fatigue_level: str = "Optimal"         # "Optimal" / "Normal" / "Yorgun" / "Tehlike"


class CognitiveMonitor:
    """EAR + PERCLOS Tabanlı Bilişsel Durum İzleyici.

    EAR formülü (Soukupová & Čech, 2016):

         |p2 - p6| + |p3 - p5|
    EAR = ─────────────────────
              2 · |p1 - p4|

    Burada:
        p1, p4 = Göz yatay köşeleri
        p2, p3 = Göz üst dikey noktaları
        p5, p6 = Göz alt dikey noktaları

    Bilişsel Yük İndeksi (Weighted Composite):
        0.35 × PERCLOS + 0.25 × kırpma_frekansı + 0.25 × kırpma_süresi + 0.15 × EAR_kararsızlığı

    Args:
        ear_threshold: EAR eşik değeri (altı = göz kapalı)
        window_sec: PERCLOS pencere boyutu (saniye)
        fps: Kamera FPS değeri
        consec_frames: Kırpma tespiti için minimum ardışık kapalı kare
    """

    # MediaPipe göz landmark indisleri (6 nokta — p1..p6)
    # Sol göz: dış köşe, üst-1, üst-2, iç köşe, alt-2, alt-1
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    # Sağ göz: iç köşe, üst-1, üst-2, dış köşe, alt-2, alt-1
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_threshold: float = 0.22,
        window_sec: int = 60,
        fps: float = 30.0,
        consec_frames: int = 2,
    ):
        self.ear_threshold = ear_threshold
        self.window_size = int(window_sec * fps)  # Ring buffer boyutu (kare)
        self.fps = fps
        self.consec_frames = consec_frames

        # Ring buffer: son window_sec saniyelik EAR değerleri
        self._ear_buffer: deque = deque(maxlen=self.window_size)

        # Kırpma durum makinesi
        self._blink_state: str = "open"  # open, closing, closed, opening
        self._closed_counter: int = 0     # Ardışık kapalı kare sayacı
        self._blink_start_time: float = 0.0
        self._blink_durations: deque = deque(maxlen=100)  # Son 100 kırpma süresi
        self._blink_timestamps: deque = deque(maxlen=100)  # Son 100 kırpma zaman damgası

        # Toplam kırpma sayacı
        self._total_blinks: int = 0

        # Başlangıç zamanı
        self._start_time: float = time.time()

    def compute_ear(self, landmarks) -> float:
        """Tek göz için EAR hesaplar.

        Args:
            landmarks: MediaPipe FaceMesh landmark listesi

        Returns:
            Her iki gözün ortalaması olan EAR değeri
        """
        if not landmarks or len(landmarks) < 468:
            return 0.3  # Varsayılan açık göz değeri

        left_ear = self._single_eye_ear(landmarks, self.LEFT_EYE)
        right_ear = self._single_eye_ear(landmarks, self.RIGHT_EYE)

        return (left_ear + right_ear) / 2.0

    def _single_eye_ear(self, landmarks, indices) -> float:
        """Tek göz EAR hesaplama.

        EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
        """
        p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
        p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
        p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
        p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
        p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
        p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

        # Dikey mesafeler
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        # Yatay mesafe
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal < 1e-6:
            return 0.3  # Sıfıra bölünme koruması

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    def update(self, landmarks) -> CognitiveState:
        """Yeni frame ile bilişsel durumu günceller.

        Args:
            landmarks: MediaPipe FaceMesh landmark listesi

        Returns:
            CognitiveState — EAR, kırpma, PERCLOS, bilişsel yük bilgisi
        """
        state = CognitiveState()

        # ── 1. EAR Hesaplama ──
        ear = self.compute_ear(landmarks)
        state.ear = ear
        self._ear_buffer.append(ear)

        # ── 2. Kırpma Tespiti (Durum Makinesi) ──
        is_eye_closed = ear < self.ear_threshold
        state.is_blinking = is_eye_closed

        current_time = time.time()

        if self._blink_state == "open":
            if is_eye_closed:
                self._closed_counter += 1
                if self._closed_counter >= self.consec_frames:
                    # Kırpma başladı
                    self._blink_state = "closed"
                    self._blink_start_time = current_time
            else:
                self._closed_counter = 0

        elif self._blink_state == "closed":
            if not is_eye_closed:
                # Kırpma bitti
                blink_duration = (current_time - self._blink_start_time) * 1000  # ms
                self._blink_durations.append(blink_duration)
                self._blink_timestamps.append(current_time)
                self._total_blinks += 1
                self._blink_state = "open"
                self._closed_counter = 0

        state.blink_count = self._total_blinks

        # ── 3. Kırpma Frekansı (kırpma/dakika) ──
        if self._blink_timestamps:
            # Son 60 saniye içindeki kırpma sayısı
            cutoff = current_time - 60.0
            recent_blinks = sum(1 for t in self._blink_timestamps if t > cutoff)
            elapsed_min = min((current_time - self._start_time) / 60.0, 1.0)
            state.blink_rate = recent_blinks / max(elapsed_min, 1/60)
        else:
            state.blink_rate = 0.0

        # ── 4. Ortalama Kırpma Süresi ──
        if self._blink_durations:
            state.avg_blink_duration_ms = float(np.mean(self._blink_durations))
        else:
            state.avg_blink_duration_ms = 0.0

        # ── 5. PERCLOS Hesaplama ──
        if len(self._ear_buffer) > 0:
            closed_frames = sum(1 for e in self._ear_buffer if e < self.ear_threshold)
            state.perclos = closed_frames / len(self._ear_buffer)
        else:
            state.perclos = 0.0

        # ── 6. Bilişsel Yük İndeksi ──
        state.cognitive_load = self._compute_cognitive_load(state)

        # ── 7. Yorgunluk Seviyesi ──
        state.fatigue_level = self._classify_fatigue(state.cognitive_load)

        return state

    def _compute_cognitive_load(self, state: CognitiveState) -> float:
        """Bilişsel yük indeksi hesaplama (weighted composite).

        Bileşenler:
            - PERCLOS (ağırlık: 0.35) — gözlerin kapalı kalma oranı
            - Kırpma frekansı (ağırlık: 0.25) — yüksek frekans = yorgunluk
            - Kırpma süresi (ağırlık: 0.25) — uzun süre = yorgunluk
            - EAR kararsızlığı (ağırlık: 0.15) — yüksek varyans = dikkat dağınıklığı
        """
        # PERCLOS normalize: [0, 0.4] → [0, 1]
        perclos_norm = min(state.perclos / 0.4, 1.0)

        # Kırpma frekansı normalize: [15, 35] → [0, 1]
        # Normal: 15-20/dakika, Yorgun: 25+/dakika
        blink_rate_norm = max(0.0, min((state.blink_rate - 15.0) / 20.0, 1.0))

        # Kırpma süresi normalize: [100, 300] ms → [0, 1]
        # Normal: 100-150ms, Yorgun: 200ms+
        duration_norm = max(0.0, min(
            (state.avg_blink_duration_ms - 100.0) / 200.0, 1.0
        ))

        # EAR kararsızlığı: son N karenin standart sapması
        if len(self._ear_buffer) >= 10:
            recent_ears = list(self._ear_buffer)[-30:]
            ear_std = float(np.std(recent_ears))
            # Normalize: [0, 0.05] → [0, 1], yüksek varyans = dikkat dağınıklığı
            ear_instability = min(ear_std / 0.05, 1.0)
        else:
            ear_instability = 0.0

        # Weighted composite
        cognitive_load = (
            0.35 * perclos_norm
            + 0.25 * blink_rate_norm
            + 0.25 * duration_norm
            + 0.15 * ear_instability
        )

        return float(np.clip(cognitive_load, 0.0, 1.0))

    @staticmethod
    def _classify_fatigue(load: float) -> str:
        """Bilişsel yük indeksinden yorgunluk seviyesi etiketi.

        Kategoriler:
            0.0 - 0.3: Optimal (rahat, odaklanmış)
            0.3 - 0.6: Normal (düzenli bilişsel aktivite)
            0.6 - 0.8: Yorgun (dikkat azalması başlıyor)
            0.8 - 1.0: Tehlike (ciddi yorgunluk, mola önerilir)
        """
        if load < 0.3:
            return "Optimal"
        elif load < 0.6:
            return "Normal"
        elif load < 0.8:
            return "Yorgun"
        else:
            return "Tehlike"

    def reset(self):
        """Tüm durumu sıfırlar."""
        self._ear_buffer.clear()
        self._blink_state = "open"
        self._closed_counter = 0
        self._blink_start_time = 0.0
        self._blink_durations.clear()
        self._blink_timestamps.clear()
        self._total_blinks = 0
        self._start_time = time.time()
