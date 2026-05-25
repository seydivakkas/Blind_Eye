import cv2
import numpy as np
import logging
from typing import Optional
from .expression_detector import ExpressionDetector
from .optical_flow_tracker import OpticalFlowTracker

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    # mediapipe kurulu olsa bile solutions API'si yoksa kullanılamaz
    _test = mp.solutions.face_mesh
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    MP_AVAILABLE = False
    logger.warning("mediapipe kullanılamıyor — ROI işleme devre dışı.")


class ROIProcessor:
    """MediaPipe FaceMesh + KLT Optik Akış Hibrit Dudak ROI Çıkarıcı.

    Tracking-by-Detection paradigması:
        - FaceMesh: Her N karede bir (ağır algılama, ~30ms)
        - KLT: Aradaki karelerde (hafif takip, ~1-2ms)

    Bu sayede 30 FPS kamera akışında düşük CPU yüküyle
    akıcı landmark takibi ve ROI çıkarma sağlanır.
    """

    LIP_OUTER = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    ]

    def __init__(self, target_size: tuple = (96, 96), margin: float = 0.25,
                 detection_interval: int = 5):
        self.target_size = target_size
        self.margin = margin
        self.face_mesh = None
        self.last_bbox = None
        self.last_landmarks = []
        self.expr_detector = ExpressionDetector()
        self.last_expressions = {
            "dominant": "Nötr",
            "confidence": 1.0,
            "scores": {"Gülümseme": 0.0, "Kaş Çatma": 0.0, "Şaşırma": 0.0}
        }

        # KLT Optik Akış Hibrit Takipçi
        self.tracker = OpticalFlowTracker(
            detection_interval=detection_interval,
            max_drift=15.0,
            fb_threshold=2.0,
        )
        self.last_tracking_quality: float = 0.0
        self.last_tracking_mode: str = "detection"  # "detection" veya "tracking"

        if MP_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Frame'den dudak ROI çıkarır.

        Hibrit yaklaşım:
        1. tracker.needs_detection → FaceMesh çalıştır, landmark'ları tracker'a ver
        2. Aksi halde → KLT ile takip et

        Returns:
            Normalize edilmiş gri tonlamalı ROI [H, W, 1] veya None
        """
        if frame is None or self.face_mesh is None:
            self._reset_state()
            return None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        landmarks = None
        use_facemesh = self.tracker.needs_detection

        # ── FaceMesh Algılama (ağır, periyodik) ──
        if use_facemesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Tracker'a yeni anchor noktaları ver
                self.tracker.update(gray, facemesh_landmarks=landmarks,
                                    frame_w=w, frame_h=h)
                self.last_tracking_mode = "detection"

                # Geometrik mimik tespiti (FaceMesh'ten tam landmark seti ile)
                self.last_expressions = self.expr_detector.detect(landmarks)
            else:
                self._reset_state()
                self.tracker.reset()
                return None
        else:
            # ── KLT Takip (hafif, her kare) ──
            self.tracker.update(gray, frame_w=w, frame_h=h)
            self.last_tracking_mode = "tracking"

        # Takip kalitesini güncelle
        self.last_tracking_quality = self.tracker.tracking_quality

        # ── Dudak noktalarından ROI hesapla ──
        lip_points = self.tracker.get_lip_points()
        if lip_points is None or len(lip_points) == 0:
            self._reset_state()
            return None

        # Piksel koordinatlarından bounding box hesapla
        xs = lip_points[:, 0]
        ys = lip_points[:, 1]

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        rx = int((np.max(xs) - np.min(xs)) / 2 * (1 + self.margin))
        ry = int((np.max(ys) - np.min(ys)) / 2 * (1 + self.margin))

        x1 = max(0, cx - rx)
        y1 = max(0, cy - ry)
        x2 = min(w, cx + rx)
        y2 = min(h, cy + ry)

        self.last_bbox = (x1, y1, x2, y2)
        self.last_landmarks = [
            (int(lip_points[i, 0]), int(lip_points[i, 1]))
            for i in range(len(lip_points))
        ]

        # ROI çıkar, normalize et
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(roi_gray, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)  # [H, W, 1]

    def _reset_state(self):
        """Durumu sıfırlar."""
        self.last_bbox = None
        self.last_landmarks = []
        self.last_tracking_quality = 0.0
        self.last_tracking_mode = "detection"
        self.last_expressions = {
            "dominant": "Nötr",
            "confidence": 1.0,
            "scores": {"Gülümseme": 0.0, "Kaş Çatma": 0.0, "Şaşırma": 0.0}
        }
