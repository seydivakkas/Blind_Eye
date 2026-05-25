"""
backend/kinematic_analyzer.py
──────────────────────────────
Zamansal Kinematik Mimik Analiz Motoru.

Yüz özelliklerinin zamana göre türevlerini hesaplayarak:
    - Hız (velocity):  v_t = (D_t - D_{t-1}) / Δt
    - İvme (acceleration): a_t = (v_t - v_{t-1}) / Δt

Bu sayede:
    - Mikro-ifade tespiti (ani ivme artışı + kısa süre)
    - Samimi vs sahte gülümseme ayrımı (Duchenne gülümseme)
    - Duygu geçiş hızı analizi

Akademik referanslar:
    - Ekman & Friesen (1978): Facial Action Coding System (FACS)
    - Yan et al. (2014): "How Fast are the Leaked Facial Expressions"
    - Cohn & Schmidt (2004): "The Timing of Facial Motion in Posed and
      Spontaneous Smiles"
"""

import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Kinematik Durum Veri Yapısı
# ═══════════════════════════════════════════════════════════════

@dataclass
class KinematicState:
    """Anlık kinematik analiz durumu."""
    velocities: Dict[str, float] = field(default_factory=dict)
    accelerations: Dict[str, float] = field(default_factory=dict)
    micro_expression: Optional[str] = None
    is_duchenne: bool = False
    emotion_transition: Optional[str] = None
    peak_velocity_channel: Optional[str] = None
    peak_acceleration_channel: Optional[str] = None


# Kinematik kanallar
CHANNELS = [
    "mouth_width",       # Ağız genişliği (Gülümseme)
    "mouth_height",      # Ağız dikey açıklığı (Şaşırma)
    "eyebrow_dist",      # İç kaş mesafesi (Kaş Çatma)
    "left_eye_height",   # Sol göz açıklığı
    "right_eye_height",  # Sağ göz açıklığı
    "lip_corner_raise",  # Dudak köşe yüksekliği
]


class KinematicAnalyzer:
    """Zamansal Kinematik Mimik Analiz Motoru.

    Ring buffer ile son N karenin feature mesafelerini tutar,
    türev (hız/ivme) hesaplar, mikro-ifade ve Duchenne tespiti yapar.

    Args:
        buffer_size: Kinematik tampon boyutu (kare sayısı)
        fps: Kamera FPS değeri (Δt hesaplama için)
        micro_accel_threshold: Mikro-ifade için ivme eşiği
        micro_min_frames: Mikro-ifade minimum süre (kare)
        micro_max_frames: Mikro-ifade maksimum süre (kare)
    """

    def __init__(
        self,
        buffer_size: int = 30,
        fps: float = 30.0,
        micro_accel_threshold: float = 0.5,
        micro_min_frames: int = 1,
        micro_max_frames: int = 6,
    ):
        self.buffer_size = buffer_size
        self.dt = 1.0 / fps  # Zaman adımı
        self.micro_accel_threshold = micro_accel_threshold
        self.micro_min_frames = micro_min_frames
        self.micro_max_frames = micro_max_frames

        # Ring buffer: her kanal için deque
        self._buffers: Dict[str, deque] = {
            ch: deque(maxlen=buffer_size) for ch in CHANNELS
        }
        self._velocities: Dict[str, deque] = {
            ch: deque(maxlen=buffer_size) for ch in CHANNELS
        }

        # Duygu geçiş takibi
        self._prev_dominant: Optional[str] = None

        # Mikro-ifade durum makinesi
        self._micro_state: str = "idle"  # idle, onset, offset
        self._micro_channel: Optional[str] = None
        self._micro_frame_count: int = 0

        # Son durum
        self._last_state = KinematicState()

    def update(self, features: Dict[str, float], dominant: str = "Nötr") -> KinematicState:
        """Yeni feature set ile kinematik durumu günceller.

        Args:
            features: {channel_name: normalized_distance, ...}
            dominant: Mevcut baskın duygu (geçiş tespiti için)

        Returns:
            KinematicState — hız, ivme, mikro-ifade, Duchenne bilgisi
        """
        state = KinematicState()

        # ── 1. Hız (Velocity) Hesaplama ──
        for ch in CHANNELS:
            val = features.get(ch, 0.0)
            self._buffers[ch].append(val)

            if len(self._buffers[ch]) >= 2:
                d_prev = self._buffers[ch][-2]
                d_curr = self._buffers[ch][-1]
                velocity = (d_curr - d_prev) / self.dt
            else:
                velocity = 0.0

            state.velocities[ch] = velocity
            self._velocities[ch].append(velocity)

        # ── 2. İvme (Acceleration) Hesaplama ──
        for ch in CHANNELS:
            if len(self._velocities[ch]) >= 2:
                v_prev = self._velocities[ch][-2]
                v_curr = self._velocities[ch][-1]
                acceleration = (v_curr - v_prev) / self.dt
            else:
                acceleration = 0.0

            state.accelerations[ch] = acceleration

        # ── 3. Peak Velocity/Acceleration Kanalları ──
        if state.velocities:
            state.peak_velocity_channel = max(
                state.velocities, key=lambda k: abs(state.velocities[k])
            )
        if state.accelerations:
            state.peak_acceleration_channel = max(
                state.accelerations, key=lambda k: abs(state.accelerations[k])
            )

        # ── 4. Mikro-İfade Tespiti ──
        state.micro_expression = self._detect_micro_expression(state)

        # ── 5. Duchenne Gülümseme Tespiti ──
        state.is_duchenne = self._detect_duchenne(state)

        # ── 6. Duygu Geçiş Tespiti ──
        if self._prev_dominant is not None and dominant != self._prev_dominant:
            state.emotion_transition = f"{self._prev_dominant} → {dominant}"
        self._prev_dominant = dominant

        self._last_state = state
        return state

    def _detect_micro_expression(self, state: KinematicState) -> Optional[str]:
        """Mikro-ifade tespiti.

        Mikro-ifade kriterleri (Yan et al., 2014):
        - Ani yüksek ivme (onset): |a| > threshold
        - Kısa süre: 40ms - 200ms (1-6 kare @ 30fps)
        - Hızlı geri dönüş (offset): ters yönde hız

        Durum makinesi:
        idle → onset (yüksek ivme) → offset (ters hız) → idle
        """
        # En yüksek ivmeli kanalı bul
        max_accel_ch = state.peak_acceleration_channel
        if max_accel_ch is None:
            return None

        max_accel = abs(state.accelerations.get(max_accel_ch, 0.0))

        if self._micro_state == "idle":
            # Ani yüksek ivme → onset
            if max_accel > self.micro_accel_threshold:
                self._micro_state = "onset"
                self._micro_channel = max_accel_ch
                self._micro_frame_count = 1
        elif self._micro_state == "onset":
            self._micro_frame_count += 1

            if self._micro_frame_count > self.micro_max_frames:
                # Çok uzun sürdü → mikro-ifade değil, normal ifade
                self._micro_state = "idle"
                self._micro_frame_count = 0
                self._micro_channel = None
            elif self._micro_frame_count >= self.micro_min_frames:
                # Ters yönde hız var mı? (offset kontrolü)
                ch = self._micro_channel
                if ch and len(self._velocities[ch]) >= 2:
                    v_prev = self._velocities[ch][-2]
                    v_curr = self._velocities[ch][-1]
                    # İşaret değişimi → offset
                    if v_prev != 0 and np.sign(v_curr) != np.sign(v_prev):
                        self._micro_state = "idle"
                        self._micro_frame_count = 0
                        detected = self._channel_to_expression(ch)
                        self._micro_channel = None
                        return detected

        return None

    def _detect_duchenne(self, state: KinematicState) -> bool:
        """Duchenne (samimi) gülümseme tespiti.

        Duchenne gülümsemesi (Cohn & Schmidt, 2004):
        - Dudak köşeleri yukarı kalkıyor (mouth_width + lip_corner_raise artıyor)
        - VE aynı anda göz çevresi kasları kasılıyor (eye_height azalıyor)

        Sahte gülümseme: sadece dudak hareketi, göz kasılması yok.
        """
        mouth_vel = state.velocities.get("mouth_width", 0.0)
        corner_vel = state.velocities.get("lip_corner_raise", 0.0)
        left_eye_vel = state.velocities.get("left_eye_height", 0.0)
        right_eye_vel = state.velocities.get("right_eye_height", 0.0)

        # Dudaklar genişliyor (pozitif hız)
        mouth_opening = mouth_vel > 0.3 or corner_vel > 0.3

        # Gözler kısılıyor (negatif hız — göz yüksekliği azalıyor)
        avg_eye_vel = (left_eye_vel + right_eye_vel) / 2.0
        eyes_squinting = avg_eye_vel < -0.1

        return mouth_opening and eyes_squinting

    @staticmethod
    def _channel_to_expression(channel: str) -> str:
        """Kinematik kanaldan ifade adına dönüştürme."""
        mapping = {
            "mouth_width": "Mikro-Gülümseme",
            "lip_corner_raise": "Mikro-Gülümseme",
            "mouth_height": "Mikro-Şaşırma",
            "eyebrow_dist": "Mikro-Kaş Çatma",
            "left_eye_height": "Mikro-Şaşırma",
            "right_eye_height": "Mikro-Şaşırma",
        }
        return mapping.get(channel, "Mikro-İfade")

    @property
    def last_state(self) -> KinematicState:
        """Son hesaplanan kinematik durum."""
        return self._last_state

    def get_velocity_history(self, channel: str, n: int = 30) -> list:
        """Belirtilen kanalın son N hız değerini döndürür (spark-line grafik için)."""
        if channel not in self._velocities:
            return []
        history = list(self._velocities[channel])
        return history[-n:]

    def reset(self):
        """Tüm tamponları sıfırlar."""
        for ch in CHANNELS:
            self._buffers[ch].clear()
            self._velocities[ch].clear()
        self._prev_dominant = None
        self._micro_state = "idle"
        self._micro_channel = None
        self._micro_frame_count = 0
        self._last_state = KinematicState()
