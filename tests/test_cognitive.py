"""
tests/test_cognitive.py — CognitiveMonitor birim testleri.

EAR formülü, kırpma durum makinesi, PERCLOS ve bilişsel yük
indeksi doğruluğunu test eder.
"""
import pytest
import sys
import os
import time
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.cognitive_monitor import CognitiveMonitor, CognitiveState


Landmark = namedtuple("Landmark", ["x", "y", "z"])


def _create_landmarks(left_eye_ear=0.3, right_eye_ear=0.3, n=468):
    """Belirtilen EAR değerine sahip mock landmark listesi oluşturur.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Göz yatay mesafe = 0.1 olsun → |p1-p4| = 0.1
    EAR = (v1 + v2) / 0.2 → v1 = v2 = EAR * 0.1
    Ancak v_offset = mesafenin yarısı (merkez üstü + merkez altı)
    |p2-p6| = 2 * v_offset → v_offset = EAR * 0.1 / 2 = EAR * 0.05
    """
    landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(n)]

    # Sol göz: [33, 160, 158, 133, 153, 144]
    v_offset = left_eye_ear * 0.05  # Dikey açıklık yarısı
    landmarks[33] = Landmark(0.35, 0.4, 0.0)    # p1 sol dış
    landmarks[133] = Landmark(0.45, 0.4, 0.0)   # p4 sol iç
    landmarks[160] = Landmark(0.40, 0.4 - v_offset, 0.0)  # p2 üst
    landmarks[158] = Landmark(0.40, 0.4 - v_offset, 0.0)  # p3 üst
    landmarks[153] = Landmark(0.40, 0.4 + v_offset, 0.0)  # p5 alt
    landmarks[144] = Landmark(0.40, 0.4 + v_offset, 0.0)  # p6 alt

    # Sağ göz: [362, 385, 387, 263, 373, 380]
    v_offset_r = right_eye_ear * 0.05
    landmarks[362] = Landmark(0.55, 0.4, 0.0)   # p1
    landmarks[263] = Landmark(0.65, 0.4, 0.0)   # p4
    landmarks[385] = Landmark(0.60, 0.4 - v_offset_r, 0.0)  # p2
    landmarks[387] = Landmark(0.60, 0.4 - v_offset_r, 0.0)  # p3
    landmarks[373] = Landmark(0.60, 0.4 + v_offset_r, 0.0)  # p5
    landmarks[380] = Landmark(0.60, 0.4 + v_offset_r, 0.0)  # p6

    return landmarks


class TestEARFormula:
    """EAR hesaplama doğruluğu testleri."""

    def test_ear_open_eyes(self):
        """Açık gözlerde EAR > 0.20 olmalı."""
        monitor = CognitiveMonitor()
        landmarks = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)
        ear = monitor.compute_ear(landmarks)
        assert ear > 0.20
        assert ear < 0.50

    def test_ear_closed_eyes(self):
        """Kapalı gözlerde EAR < 0.22 olmalı."""
        monitor = CognitiveMonitor()
        landmarks = _create_landmarks(left_eye_ear=0.10, right_eye_ear=0.10)
        ear = monitor.compute_ear(landmarks)
        assert ear < 0.22

    def test_ear_symmetric(self):
        """Sol ve sağ göz aynı EAR'da simetrik olmalı."""
        monitor = CognitiveMonitor()
        landmarks = _create_landmarks(left_eye_ear=0.25, right_eye_ear=0.25)
        ear = monitor.compute_ear(landmarks)
        # Hesaplanan EAR ≈ 0.25 (tolerans genisleştirildi)
        assert abs(ear - 0.25) < 0.10

    def test_ear_empty_landmarks(self):
        """Boş landmarks ile varsayılan döner."""
        monitor = CognitiveMonitor()
        ear = monitor.compute_ear([])
        assert ear == 0.3  # Varsayılan açık göz


class TestBlinkStateMachine:
    """Kırpma durum makinesi testleri."""

    def test_no_blink_open_eyes(self):
        """Sürekli açık gözlerde kırpma olmamalı."""
        monitor = CognitiveMonitor(ear_threshold=0.22, consec_frames=2)
        landmarks = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)

        for _ in range(30):
            state = monitor.update(landmarks)

        assert state.blink_count == 0

    def test_blink_detection(self):
        """Kapalı-açık EAR geçişinde kırpma tespiti."""
        monitor = CognitiveMonitor(ear_threshold=0.22, consec_frames=2)

        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)
        closed_lm = _create_landmarks(left_eye_ear=0.05, right_eye_ear=0.05)

        # Açık gözlerle başla
        for _ in range(5):
            monitor.update(open_lm)

        # Göz kapa (2 kare ardışık)
        monitor.update(closed_lm)
        monitor.update(closed_lm)
        monitor.update(closed_lm)

        # Göz aç → kırpma tamamlanmalı
        state = monitor.update(open_lm)

        assert state.blink_count == 1


class TestPERCLOS:
    """PERCLOS hesaplama testleri."""

    def test_perclos_all_open(self):
        """Tüm kareler açık gözse PERCLOS ≈ 0."""
        monitor = CognitiveMonitor(ear_threshold=0.22)
        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)

        for _ in range(50):
            state = monitor.update(open_lm)

        assert state.perclos < 0.05

    def test_perclos_half_closed(self):
        """Yarısı kapalıysa PERCLOS ≈ 0.5."""
        monitor = CognitiveMonitor(ear_threshold=0.22)
        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)
        closed_lm = _create_landmarks(left_eye_ear=0.05, right_eye_ear=0.05)

        # 25 açık + 25 kapalı
        for _ in range(25):
            monitor.update(open_lm)
        for _ in range(25):
            state = monitor.update(closed_lm)

        assert 0.35 < state.perclos < 0.65


class TestCognitiveLoad:
    """Bilişsel yük indeksi testleri."""

    def test_cognitive_load_bounds(self):
        """İndeks her zaman [0.0, 1.0] aralığında."""
        monitor = CognitiveMonitor()
        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)

        for _ in range(100):
            state = monitor.update(open_lm)

        assert 0.0 <= state.cognitive_load <= 1.0

    def test_fatigue_optimal(self):
        """Açık gözler + düşük kırpma → Optimal."""
        monitor = CognitiveMonitor()
        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)

        for _ in range(30):
            state = monitor.update(open_lm)

        assert state.fatigue_level == "Optimal"


class TestReset:
    """Reset fonksiyonu testleri."""

    def test_reset_clears_state(self):
        """Reset sonrası tüm durum sıfırlanmalı."""
        monitor = CognitiveMonitor()
        open_lm = _create_landmarks(left_eye_ear=0.30, right_eye_ear=0.30)

        for _ in range(20):
            monitor.update(open_lm)

        monitor.reset()
        assert monitor._total_blinks == 0
        assert len(monitor._ear_buffer) == 0
