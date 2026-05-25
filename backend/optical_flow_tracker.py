"""
backend/optical_flow_tracker.py
────────────────────────────────
KLT Optik Akış + FaceMesh Hibrit Landmark Takipçisi.

Tracking-by-Detection paradigması (Kalal et al., 2012):
    - FaceMesh: Her N karede bir (ağır algılama, ~30ms)
    - KLT: Aradaki karelerde (hafif takip, ~1-2ms)

Forward-backward hata kontrolü ile takip kalitesini ölçer.
Drift eşiği aşıldığında otomatik re-detection tetikler.

Akademik referanslar:
    - Lucas & Kanade (1981): "An Iterative Image Registration Technique"
    - Bouguet (2001): Pyramidal LK Optical Flow
    - Kalal et al. (2012): Tracking-Learning-Detection
"""

import numpy as np
import cv2
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class OpticalFlowTracker:
    """KLT Optik Akış + FaceMesh Hibrit Landmark Takipçisi.

    Attributes:
        detection_interval: FaceMesh kaç karede bir çalışır
        max_drift: Bu pikselden fazla kayma olursa re-detection zorlanır
        fb_threshold: Forward-backward hata eşiği (piksel)
    """

    # Takip edilecek yüz noktası indisleri
    # Dudak dış sınırı (20 nokta)
    LIP_OUTER = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]
    # Göz noktaları (8 nokta — sol ve sağ göz köşeleri + dikey eksen)
    EYE_POINTS = [33, 159, 133, 145, 263, 386, 362, 374]
    # Kaş iç uçları (4 nokta)
    BROW_POINTS = [70, 107, 336, 300]

    # Lucas-Kanade parametreleri
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(
        self,
        detection_interval: int = 5,
        max_drift: float = 15.0,
        fb_threshold: float = 2.0,
    ):
        self.detection_interval = detection_interval
        self.max_drift = max_drift
        self.fb_threshold = fb_threshold

        # İç durum
        self._frame_count: int = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._anchor_points: Optional[np.ndarray] = None  # FaceMesh'ten gelen orijinal pozisyonlar
        self._tracked_points: Optional[np.ndarray] = None  # KLT ile güncellenen pozisyonlar
        self._point_status: Optional[np.ndarray] = None     # Her noktanın takip durumu (1=iyi, 0=kayıp)
        self._tracking_quality: float = 0.0
        self._force_detection: bool = True  # İlk karede detection zorunlu

        # Takip edilen toplam nokta sayısı
        self._all_indices: List[int] = self.LIP_OUTER + self.EYE_POINTS + self.BROW_POINTS
        self._n_points = len(self._all_indices)

        logger.info(
            f"OpticalFlowTracker: interval={detection_interval}, "
            f"max_drift={max_drift}px, {self._n_points} nokta takip edilecek"
        )

    @property
    def needs_detection(self) -> bool:
        """FaceMesh çalıştırılması gerekiyor mu?

        True döner eğer:
        - İlk kare (henüz anchor yok)
        - detection_interval doldu
        - Drift eşiği aşıldı
        - Force detection flag'i set edilmişse
        """
        if self._force_detection or self._anchor_points is None:
            return True
        if self._frame_count % self.detection_interval == 0:
            return True
        return False

    @property
    def tracking_quality(self) -> float:
        """Takip kalitesi [0.0 - 1.0].

        KLT status değerlerinin ortalaması × drift ceza faktörü.
        """
        return self._tracking_quality

    @property
    def tracked_points(self) -> Optional[np.ndarray]:
        """Güncel takip edilen noktalar [N, 2] float32."""
        return self._tracked_points

    @property
    def is_tracking(self) -> bool:
        """KLT takip modunda mı (True) yoksa FaceMesh algılama modunda mı (False)."""
        return (
            self._tracked_points is not None
            and not self.needs_detection
        )

    def get_lip_points(self) -> Optional[np.ndarray]:
        """Sadece dudak noktalarını döndürür [20, 2]."""
        if self._tracked_points is None:
            return None
        n_lip = len(self.LIP_OUTER)
        return self._tracked_points[:n_lip].copy()

    def get_eye_points(self) -> Optional[np.ndarray]:
        """Sadece göz noktalarını döndürür [8, 2]."""
        if self._tracked_points is None:
            return None
        n_lip = len(self.LIP_OUTER)
        n_eye = len(self.EYE_POINTS)
        return self._tracked_points[n_lip:n_lip + n_eye].copy()

    def get_brow_points(self) -> Optional[np.ndarray]:
        """Sadece kaş noktalarını döndürür [4, 2]."""
        if self._tracked_points is None:
            return None
        n_lip = len(self.LIP_OUTER)
        n_eye = len(self.EYE_POINTS)
        return self._tracked_points[n_lip + n_eye:].copy()

    def update(
        self,
        frame_gray: np.ndarray,
        facemesh_landmarks=None,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> Optional[np.ndarray]:
        """Ana güncelleme fonksiyonu.

        Args:
            frame_gray: Gri tonlamalı kare [H, W] uint8
            facemesh_landmarks: FaceMesh landmark listesi (verilmişse anchor güncellenir)
            frame_w: Frame genişliği (landmark piksel dönüşümü için)
            frame_h: Frame yüksekliği

        Returns:
            Güncel nokta pozisyonları [N, 2] float32 veya None
        """
        self._frame_count += 1

        # ── FaceMesh'ten yeni anchor noktaları geldi ──
        if facemesh_landmarks is not None:
            self._set_anchors(facemesh_landmarks, frame_w, frame_h)
            self._prev_gray = frame_gray.copy()
            self._force_detection = False
            self._tracking_quality = 1.0
            return self._tracked_points

        # ── KLT ile takip ──
        if self._prev_gray is not None and self._tracked_points is not None:
            new_points = self._track_klt(self._prev_gray, frame_gray)
            if new_points is not None:
                self._tracked_points = new_points
                self._compute_quality()
            else:
                # KLT tamamen başarısız → re-detection zorla
                self._force_detection = True
                self._tracking_quality = 0.0

        self._prev_gray = frame_gray.copy()
        return self._tracked_points

    def _set_anchors(self, landmarks, frame_w: int, frame_h: int):
        """FaceMesh landmark'larından anchor noktaları oluşturur."""
        points = []
        for idx in self._all_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px = float(lm.x * frame_w)
                py = float(lm.y * frame_h)
                points.append([px, py])
            else:
                # Geçersiz indis — varsayılan pozisyon
                points.append([frame_w / 2.0, frame_h / 2.0])

        self._anchor_points = np.array(points, dtype=np.float32)
        self._tracked_points = self._anchor_points.copy()
        self._point_status = np.ones(self._n_points, dtype=np.uint8)

    def _track_klt(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> Optional[np.ndarray]:
        """Lucas-Kanade optik akış ile noktaları takip eder.

        Forward-backward hata kontrolü uygular:
        1. prev → curr: İleri takip
        2. curr → prev: Geri takip
        3. İleri başlangıç ile geri bitiş arasındaki mesafe < eşik ise güvenilir
        """
        prev_pts = self._tracked_points.reshape(-1, 1, 2)

        # ── Forward pass: prev → curr ──
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.LK_PARAMS
        )

        if next_pts is None or status_fwd is None:
            return None

        # ── Backward pass: curr → prev (forward-backward check) ──
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, next_pts, None, **self.LK_PARAMS
        )

        if back_pts is None or status_bwd is None:
            return None

        # ── Forward-backward hata hesaplama ──
        fb_error = np.linalg.norm(
            prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
        )

        # Durumu güncelle: her iki yönde de başarılı VE FB hatası düşük
        status_combined = (
            (status_fwd.flatten() == 1)
            & (status_bwd.flatten() == 1)
            & (fb_error < self.fb_threshold)
        ).astype(np.uint8)

        self._point_status = status_combined

        # Güncel noktaları oluştur
        result = next_pts.reshape(-1, 2).copy()

        # Başarısız noktaları son bilinen pozisyonda tut
        failed_mask = status_combined == 0
        result[failed_mask] = self._tracked_points[failed_mask]

        return result

    def _compute_quality(self):
        """Takip kalitesini hesaplar.

        quality = (başarılı nokta oranı) × (1 - normalized_drift)
        """
        if self._point_status is None or self._anchor_points is None:
            self._tracking_quality = 0.0
            return

        # 1. Başarılı nokta oranı
        success_ratio = float(np.mean(self._point_status))

        # 2. Ortalama drift (anchor'dan kayma)
        drift = np.linalg.norm(
            self._tracked_points - self._anchor_points, axis=1
        )
        mean_drift = float(np.mean(drift))

        # Drift ceza faktörü (sigmoid benzeri)
        drift_penalty = 1.0 / (1.0 + mean_drift / self.max_drift)

        self._tracking_quality = float(np.clip(
            success_ratio * drift_penalty, 0.0, 1.0
        ))

        # Drift çok yüksekse re-detection zorla
        if mean_drift > self.max_drift:
            self._force_detection = True
            logger.debug(
                f"Drift eşiği aşıldı ({mean_drift:.1f} > {self.max_drift}px) "
                f"→ re-detection zorunlu"
            )

    def reset(self):
        """Takipçi durumunu sıfırlar."""
        self._frame_count = 0
        self._prev_gray = None
        self._anchor_points = None
        self._tracked_points = None
        self._point_status = None
        self._tracking_quality = 0.0
        self._force_detection = True
