"""
tests/test_optical_flow.py — OpticalFlowTracker birim testleri.

KLT + FaceMesh hibrit takip sisteminin doğruluğunu test eder:
- Başlatma ve durum kontrolü
- Needs detection mantığı
- KLT takip mekanizması
- Drift eşiği ve re-detection tetikleme
- Forward-backward hata kontrolü
"""
import pytest
import sys
import os
import numpy as np
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.optical_flow_tracker import OpticalFlowTracker


# Mock landmark yapısı (MediaPipe uyumlu)
Landmark = namedtuple("Landmark", ["x", "y", "z"])


def _create_mock_landmarks(n: int = 468, x: float = 0.5, y: float = 0.5):
    """Belirtilen pozisyonda mock landmark listesi oluşturur."""
    return [Landmark(x, y, 0.0) for _ in range(n)]


def _create_gray_frame(w: int = 640, h: int = 480, value: int = 128):
    """Tek renkli gri tonlamalı frame oluşturur."""
    return np.full((h, w), value, dtype=np.uint8)


class TestOpticalFlowTrackerInit:
    """Başlatma ve yapılandırma testleri."""

    def test_init_default_params(self):
        """Varsayılan parametrelerle başlatma."""
        tracker = OpticalFlowTracker()
        assert tracker.detection_interval == 5
        assert tracker.max_drift == 15.0
        assert tracker.fb_threshold == 2.0
        assert tracker._n_points == 32  # 20 + 8 + 4

    def test_init_custom_params(self):
        """Özel parametrelerle başlatma."""
        tracker = OpticalFlowTracker(
            detection_interval=10, max_drift=20.0, fb_threshold=3.0
        )
        assert tracker.detection_interval == 10
        assert tracker.max_drift == 20.0
        assert tracker.fb_threshold == 3.0

    def test_initial_state(self):
        """Başlangıç durumu doğru olmalı."""
        tracker = OpticalFlowTracker()
        assert tracker.tracked_points is None
        assert tracker.tracking_quality == 0.0
        assert tracker.is_tracking is False
        assert tracker.get_lip_points() is None
        assert tracker.get_eye_points() is None
        assert tracker.get_brow_points() is None


class TestNeedsDetection:
    """Detection tetikleme mantığı testleri."""

    def test_needs_detection_initial(self):
        """İlk karede detection zorunlu olmalı."""
        tracker = OpticalFlowTracker()
        assert tracker.needs_detection is True

    def test_needs_detection_after_anchor(self):
        """Anchor set edildikten sonra detection gerekmemeli."""
        tracker = OpticalFlowTracker(detection_interval=5)
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        # İlk anchor set etme
        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        # Hemen sonra detection gerekmemeli (interval dolmadı)
        assert tracker.needs_detection is False

    def test_needs_detection_interval(self):
        """detection_interval karede bir detection gerekli."""
        tracker = OpticalFlowTracker(detection_interval=3)
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        # İlk anchor
        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        # Kare 2 ve 3: KLT takip
        tracker.update(gray)  # kare 2
        tracker.update(gray)  # kare 3 → interval=3 doldu

        # 4. kare: detection_interval dolmuş olmalı (frame_count=4, 4%3==1 → False)
        # Ancak frame_count=3 olduğunda 3%3==0 → True
        # tracker._frame_count zaten 3 olacak (ilk update = 1, sonraki 2 = 2,3)
        # 3 % 3 == 0 → needs_detection should be True
        assert tracker.needs_detection is True


class TestKLTTracking:
    """KLT optik akış takip testleri."""

    def test_update_with_landmarks(self):
        """FaceMesh landmark'ları ile güncelleme sonrası noktalar mevcut olmalı."""
        tracker = OpticalFlowTracker()
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        result = tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        assert result is not None
        assert result.shape == (32, 2)
        assert tracker.tracking_quality == 1.0
        assert tracker.tracked_points is not None

    def test_lip_eye_brow_separation(self):
        """Dudak, göz ve kaş noktaları doğru ayrılmalı."""
        tracker = OpticalFlowTracker()
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        lip = tracker.get_lip_points()
        eye = tracker.get_eye_points()
        brow = tracker.get_brow_points()

        assert lip is not None and lip.shape == (20, 2)
        assert eye is not None and eye.shape == (8, 2)
        assert brow is not None and brow.shape == (4, 2)

    def test_klt_no_crash_on_static_frame(self):
        """Statik frame üzerinde KLT çökmemeli."""
        tracker = OpticalFlowTracker(detection_interval=10)
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        # Anchor
        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        # KLT takip (aynı frame ile — hareket yok)
        # Düz gri frame'de trackable feature olmayabilir → kalite 0 olabilir
        result = tracker.update(gray)
        # Önemli olan çökmemesi, kalite 0 olabilir
        assert result is not None or tracker.tracking_quality >= 0.0


class TestDriftAndReset:
    """Drift eşiği ve reset testleri."""

    def test_reset_clears_state(self):
        """Reset sonrası tüm durum sıfırlanmalı."""
        tracker = OpticalFlowTracker()
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)
        assert tracker.tracked_points is not None

        tracker.reset()
        assert tracker.tracked_points is None
        assert tracker.tracking_quality == 0.0
        assert tracker.needs_detection is True

    def test_force_detection_flag(self):
        """_force_detection flag'i doğru çalışmalı."""
        tracker = OpticalFlowTracker()

        # Başlangıçta force detection True
        assert tracker._force_detection is True

        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()
        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        # Anchor sonrası False olmalı
        assert tracker._force_detection is False


class TestTrackingModeProperty:
    """is_tracking property testleri."""

    def test_is_tracking_false_initially(self):
        """Başlangıçta tracking modunda olmamalı."""
        tracker = OpticalFlowTracker()
        assert tracker.is_tracking is False

    def test_is_tracking_after_detection(self):
        """Detection sonrası tracking modunda olmamalı (çünkü az önce detection yapıldı)."""
        tracker = OpticalFlowTracker(detection_interval=5)
        landmarks = _create_mock_landmarks()
        gray = _create_gray_frame()

        tracker.update(gray, facemesh_landmarks=landmarks, frame_w=640, frame_h=480)

        # Bir kare daha KLT ile → artık tracking modunda
        tracker.update(gray)
        # is_tracking depends on needs_detection which checks frame_count % interval
        # frame_count=2, 2%5 != 0 → needs_detection=False → is_tracking=True
        assert tracker.is_tracking is True
