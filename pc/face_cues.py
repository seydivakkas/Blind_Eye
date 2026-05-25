"""
pc/face_cues.py
═══════════════
Yüz İpuçları → Bağlam / Punctuation Sinyalleri

MediaPipe FaceMesh landmark'larından aşağıdaki ipuçlarını çıkarır:
    1. Kaş kalkışı (eyebrow raise) → soru işareti (?) ipucu
    2. Göz kırpma (blink / long blink) → nokta (.) / cümle sonu ipucu
    3. Baş eğimi (head tilt) → soru tonlaması
    4. Baş nod (head nod) → onaylama / virgül bağlamı

Bu sinyaller fusion_decoder.py'ye iletilir ve CTC decode sırasında
punctuation yerleştirme ve bağlam düzeltme için kullanılır.

Referans:
    - Ekman & Friesen (1978): Facial Action Coding System (FACS)
    - Soukupová & Čech (2016): Eye Aspect Ratio (EAR) blink detection
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ── Landmark indisleri ──
# Göz (EAR hesaplama için 6 nokta)
_LEFT_EYE = [33, 160, 158, 133, 153, 144]    # p1-p6
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Kaş iç uçları (kaş çatma/kalkış)
_LEFT_EYEBROW_INNER = 107
_RIGHT_EYEBROW_INNER = 336
_LEFT_EYEBROW_OUTER = 70
_RIGHT_EYEBROW_OUTER = 300

# Göz dış kenarları (normalizasyon skalası)
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263

# Baş pozisyonu (tilt/nod)
_NOSE_TIP = 1
_FOREHEAD = 10
_CHIN = 152
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


@dataclass
class FaceCueResult:
    """Tek frame'in yüz ipucu çıktısı."""
    # Kaş
    eyebrow_raised: bool = False
    eyebrow_raise_score: float = 0.0      # 0.0–1.0

    # Göz kırpma
    is_blinking: bool = False
    blink_duration_ms: float = 0.0
    long_blink: bool = False               # >300ms = cümle sonu ipucu
    ear: float = 0.0                        # Eye Aspect Ratio

    # Baş hareketi
    head_tilt: float = 0.0                 # Derece (-: sola, +: sağa)
    head_nod_detected: bool = False        # Dikey sallanma
    head_shake_detected: bool = False      # Yatay sallanma

    # Türetilmiş punctuation sinyalleri
    suggest_question: bool = False         # Kaş kalkışı + baş eğimi → ?
    suggest_period: bool = False           # Uzun göz kapanması → .
    suggest_comma: bool = False            # Kısa nod → ,
    suggest_exclamation: bool = False      # Kaş çatma + nod → !

    # Meta
    confidence: float = 0.0
    latency_ms: float = 0.0


class FaceCueAnalyzer:
    """Yüz ipuçları analiz motoru.

    Parameters
    ----------
    fps : float
        Video stream FPS'i (temporal hesaplamalar için).
    ear_threshold : float
        EAR eşik değeri — altı = göz kapalı.
    blink_consec_frames : int
        Kırpma için minimum ardışık kapalı frame sayısı.
    long_blink_ms : float
        Uzun kırpma eşiği (ms) — cümle sonu ipucu.
    eyebrow_raise_threshold : float
        Kaş kalkışı algılama eşiği (normalize edilmiş mesafe).
    nod_buffer_size : int
        Baş hareketi analizi için ring buffer boyutu.
    nod_threshold : float
        Nod algılama eşiği (piksel farkı).
    """

    def __init__(
        self,
        fps: float = 15.0,
        ear_threshold: float = 0.21,
        blink_consec_frames: int = 2,
        long_blink_ms: float = 300.0,
        eyebrow_raise_threshold: float = 0.06,
        nod_buffer_size: int = 20,
        nod_threshold: float = 0.008,
    ):
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.long_blink_ms = long_blink_ms
        self.eyebrow_raise_threshold = eyebrow_raise_threshold
        self.nod_threshold = nod_threshold

        # Blink tracking
        self._blink_counter = 0
        self._blink_start_time: Optional[float] = None

        # Eyebrow baseline (kalibrasyon ile güncellenir)
        self._eyebrow_baseline: Optional[float] = None
        self._eyebrow_calibration_buffer: deque = deque(maxlen=60)

        # Head movement ring buffers
        self._pitch_buffer: deque = deque(maxlen=nod_buffer_size)
        self._yaw_buffer: deque = deque(maxlen=nod_buffer_size)

    def analyze(self, landmarks: list) -> FaceCueResult:
        """Landmark listesinden yüz ipuçlarını çıkar.

        Parameters
        ----------
        landmarks : list[tuple(x, y)]
            478 piksel koordinat (preprocess.py'den).

        Returns
        -------
        FaceCueResult
            Tüm yüz ipuçları ve türetilmiş punctuation sinyalleri.
        """
        t0 = time.perf_counter()
        result = FaceCueResult()

        if landmarks is None or len(landmarks) < 468:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Normalizasyon skalası (göz arası mesafe)
        left_eye_outer = np.array(landmarks[_LEFT_EYE_OUTER], dtype=np.float32)
        right_eye_outer = np.array(landmarks[_RIGHT_EYE_OUTER], dtype=np.float32)
        eye_dist = float(np.linalg.norm(right_eye_outer - left_eye_outer))
        if eye_dist < 1.0:
            eye_dist = 1.0

        # ═══════════════════════════════════════
        #  1. EAR + BLINK
        # ═══════════════════════════════════════
        ear_left = self._compute_ear(landmarks, _LEFT_EYE)
        ear_right = self._compute_ear(landmarks, _RIGHT_EYE)
        result.ear = (ear_left + ear_right) / 2.0

        eyes_closed = result.ear < self.ear_threshold

        if eyes_closed:
            self._blink_counter += 1
            if self._blink_start_time is None:
                self._blink_start_time = time.time()
        else:
            if self._blink_counter >= self.blink_consec_frames:
                result.is_blinking = True
                if self._blink_start_time is not None:
                    result.blink_duration_ms = (
                        time.time() - self._blink_start_time
                    ) * 1000
                    result.long_blink = result.blink_duration_ms >= self.long_blink_ms
            self._blink_counter = 0
            self._blink_start_time = None

        # ═══════════════════════════════════════
        #  2. KAŞ KALKIŞI
        # ═══════════════════════════════════════
        brow_left = np.array(landmarks[_LEFT_EYEBROW_INNER], dtype=np.float32)
        brow_right = np.array(landmarks[_RIGHT_EYEBROW_INNER], dtype=np.float32)
        eye_left_center = np.array(landmarks[159], dtype=np.float32)   # sol göz üst
        eye_right_center = np.array(landmarks[386], dtype=np.float32)  # sağ göz üst

        # Kaş–göz dikey mesafesi (normalize)
        brow_height_left = (eye_left_center[1] - brow_left[1]) / eye_dist
        brow_height_right = (eye_right_center[1] - brow_right[1]) / eye_dist
        avg_brow_height = (brow_height_left + brow_height_right) / 2.0

        # Kalibrasyon (ilk 60 frame nötr baseline olarak alınır)
        self._eyebrow_calibration_buffer.append(avg_brow_height)
        if self._eyebrow_baseline is None and len(self._eyebrow_calibration_buffer) >= 30:
            self._eyebrow_baseline = float(np.median(self._eyebrow_calibration_buffer))

        if self._eyebrow_baseline is not None:
            brow_delta = avg_brow_height - self._eyebrow_baseline
            result.eyebrow_raise_score = float(np.clip(
                brow_delta / self.eyebrow_raise_threshold, 0.0, 1.0
            ))
            result.eyebrow_raised = brow_delta > self.eyebrow_raise_threshold

        # ═══════════════════════════════════════
        #  3. BAŞ EĞİMİ (TILT) + NOD
        # ═══════════════════════════════════════
        nose = np.array(landmarks[_NOSE_TIP], dtype=np.float32)
        forehead = np.array(landmarks[_FOREHEAD], dtype=np.float32)
        chin = np.array(landmarks[_CHIN], dtype=np.float32)
        left_cheek = np.array(landmarks[_LEFT_CHEEK], dtype=np.float32)
        right_cheek = np.array(landmarks[_RIGHT_CHEEK], dtype=np.float32)

        # Pitch (dikey — nod) = burun-alın vektörünün dikey bileşeni
        face_vertical = forehead - chin
        pitch = float(nose[1] - forehead[1]) / eye_dist  # Normalize
        self._pitch_buffer.append(pitch)

        # Yaw (yatay — shake) = burun ile yüz merkezi farkı
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2.0
        yaw = float(nose[0] - face_center_x) / eye_dist
        self._yaw_buffer.append(yaw)

        # Tilt (eğim açısı)
        dx = right_cheek[0] - left_cheek[0]
        dy = right_cheek[1] - left_cheek[1]
        result.head_tilt = float(np.degrees(np.arctan2(dy, dx)))

        # Nod algılama (pitch buffer'da min-max farkı)
        if len(self._pitch_buffer) >= 10:
            pitch_arr = np.array(self._pitch_buffer)
            pitch_range = float(np.max(pitch_arr[-10:]) - np.min(pitch_arr[-10:]))
            result.head_nod_detected = pitch_range > self.nod_threshold

        # Shake algılama
        if len(self._yaw_buffer) >= 10:
            yaw_arr = np.array(self._yaw_buffer)
            yaw_range = float(np.max(yaw_arr[-10:]) - np.min(yaw_arr[-10:]))
            result.head_shake_detected = yaw_range > self.nod_threshold * 1.5

        # ═══════════════════════════════════════
        #  4. PUNCTUATION SİNYALLERİ
        # ═══════════════════════════════════════
        # Soru işareti: kaş kalkışı + baş eğimi
        result.suggest_question = (
            result.eyebrow_raised and abs(result.head_tilt) > 3.0
        )

        # Nokta: uzun göz kapanması
        result.suggest_period = result.long_blink

        # Virgül: kısa nod (onaylama)
        result.suggest_comma = (
            result.head_nod_detected and not result.eyebrow_raised
        )

        # Ünlem: kaş çatma + güçlü nod (vurgu)
        result.suggest_exclamation = (
            result.head_nod_detected
            and not result.eyebrow_raised
            and len(self._pitch_buffer) >= 10
            and float(np.max(np.array(self._pitch_buffer)[-10:])
                       - np.min(np.array(self._pitch_buffer)[-10:]))
            > self.nod_threshold * 2.5
        )

        # Genel güven skoru
        result.confidence = min(1.0, eye_dist / 100.0)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def reset(self):
        """Tüm state'leri sıfırla."""
        self._blink_counter = 0
        self._blink_start_time = None
        self._eyebrow_baseline = None
        self._eyebrow_calibration_buffer.clear()
        self._pitch_buffer.clear()
        self._yaw_buffer.clear()

    @staticmethod
    def _compute_ear(landmarks: list, eye_indices: list) -> float:
        """Eye Aspect Ratio (Soukupová & Čech, 2016).

        EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
        """
        p = [np.array(landmarks[i], dtype=np.float32) for i in eye_indices]
        v1 = np.linalg.norm(p[1] - p[5])  # |p2 - p6|
        v2 = np.linalg.norm(p[2] - p[4])  # |p3 - p5|
        h = np.linalg.norm(p[0] - p[3])   # |p1 - p4|
        if h < 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))
