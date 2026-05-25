"""
pc/preprocess.py
════════════════
MediaPipe FaceMesh → Dudak ROI (96×96) + Landmark Delta

Görevler:
    1. MediaPipe FaceMesh ile yüz landmark'larını çıkar
    2. Dudak bölgesini 96×96 grayscale ROI olarak kırp
    3. Frame-to-frame landmark delta hesapla (temporal feature)
    4. Tüm landmark'ları face_cues.py'ye ilet

Giriş:  BGR frame [H, W, 3] uint8
Çıkış:  PreprocessResult dataclass
        - roi        : np.ndarray [96, 96, 1] float32 (normalize)
        - landmarks  : list[tuple(x,y)] — 478 nokta piksel koordinatları
        - lip_delta  : np.ndarray [20, 2] float32 — dudak noktası farkları
        - bbox       : tuple (x1, y1, x2, y2) — dudak bounding box

Referans:
    - Google MediaPipe FaceMesh (468+10 iris = 478 landmark)
    - LIP_OUTER: 20 noktalık dış dudak konturu
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("mediapipe yüklü değil — preprocess devre dışı.")


# MediaPipe FaceMesh dudak dış kontur indisleri (20 nokta)
LIP_OUTER_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Göz landmark'ları (EAR hesaplama ve face_cues için)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Kaş iç uçları
LEFT_EYEBROW_INNER = 107
RIGHT_EYEBROW_INNER = 336

# Burun ucu ve alın (baş eğimi/nod hesaplama)
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


@dataclass
class PreprocessResult:
    """Tek bir frame'in preprocessing çıktısı."""
    roi: Optional[np.ndarray] = None            # [96, 96, 1] float32
    landmarks: Optional[list] = None            # [(x,y), ...] 478 nokta
    lip_landmarks: Optional[np.ndarray] = None  # [20, 2] float32 piksel
    lip_delta: Optional[np.ndarray] = None      # [20, 2] float32 frame farkı
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    face_detected: bool = False
    latency_ms: float = 0.0


class Preprocessor:
    """MediaPipe FaceMesh tabanlı dudak ROI çıkarıcı + landmark delta hesaplayıcı.

    Parameters
    ----------
    roi_size : tuple
        Çıktı ROI boyutu (genişlik, yükseklik). Varsayılan (96, 96).
    margin : float
        ROI etrafındaki boşluk oranı. 0.25 = %25 genişletme.
    refine_landmarks : bool
        True ise iris landmark'larını da çıkar (478 nokta).
    """

    def __init__(
        self,
        roi_size: Tuple[int, int] = (96, 96),
        margin: float = 0.25,
        refine_landmarks: bool = True,
    ):
        self.roi_size = roi_size
        self.margin = margin

        self._face_mesh = None
        self._prev_lip_pts: Optional[np.ndarray] = None

        if _MP_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("Preprocessor hazır (MediaPipe FaceMesh)")
        else:
            logger.warning("Preprocessor: MediaPipe yok — mock mod aktif")

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """Tek frame'i işle.

        Parameters
        ----------
        frame : np.ndarray
            BGR [H, W, 3] uint8 formatında video frame.

        Returns
        -------
        PreprocessResult
            ROI, landmarks, delta ve bbox bilgisi.
        """
        t0 = time.perf_counter()
        result = PreprocessResult()

        if frame is None or self._face_mesh is None:
            return result

        h, w = frame.shape[:2]

        # ── MediaPipe FaceMesh ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self._face_mesh.process(rgb)

        if not mp_result.multi_face_landmarks:
            self._prev_lip_pts = None
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        face_lm = mp_result.multi_face_landmarks[0].landmark
        result.face_detected = True

        # ── Tüm landmark'ları piksel koordinatına çevir ──
        result.landmarks = [
            (int(lm.x * w), int(lm.y * h)) for lm in face_lm
        ]

        # ── Dudak noktaları (piksel) ──
        lip_pts = np.array(
            [(face_lm[i].x * w, face_lm[i].y * h) for i in LIP_OUTER_IDX],
            dtype=np.float32,
        )
        result.lip_landmarks = lip_pts

        # ── Landmark Delta (frame-to-frame) ──
        if self._prev_lip_pts is not None:
            # Göz arası mesafe ile normalize et
            left_eye = np.array(
                [face_lm[33].x * w, face_lm[33].y * h], dtype=np.float32
            )
            right_eye = np.array(
                [face_lm[263].x * w, face_lm[263].y * h], dtype=np.float32
            )
            eye_dist = np.linalg.norm(right_eye - left_eye)
            if eye_dist < 1.0:
                eye_dist = 1.0

            raw_delta = lip_pts - self._prev_lip_pts
            result.lip_delta = raw_delta / eye_dist  # Normalize edilmiş delta
        else:
            result.lip_delta = np.zeros((len(LIP_OUTER_IDX), 2), dtype=np.float32)

        self._prev_lip_pts = lip_pts.copy()

        # ── Dudak Bounding Box + ROI ──
        xs = lip_pts[:, 0]
        ys = lip_pts[:, 1]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        rx = float((np.max(xs) - np.min(xs)) / 2 * (1 + self.margin))
        ry = float((np.max(ys) - np.min(ys)) / 2 * (1 + self.margin))

        x1 = max(0, int(cx - rx))
        y1 = max(0, int(cy - ry))
        x2 = min(w, int(cx + rx))
        y2 = min(h, int(cy + ry))
        result.bbox = (x1, y1, x2, y2)

        # ROI kırp + normalize
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, self.roi_size)
            roi_norm = roi_resized.astype(np.float32) / 255.0
            result.roi = np.expand_dims(roi_norm, axis=-1)  # [96, 96, 1]

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def reset(self):
        """Delta state'i sıfırla (sahne değişikliğinde)."""
        self._prev_lip_pts = None

    def close(self):
        """Kaynakları serbest bırak."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
