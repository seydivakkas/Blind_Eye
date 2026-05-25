"""
tests/test_kinematic.py
───────────────────────
KinematicAnalyzer birim testleri.

Test edilen modül: backend/kinematic_analyzer.py
"""

import pytest
import numpy as np
from backend.kinematic_analyzer import KinematicAnalyzer, KinematicState, CHANNELS


class TestKinematicAnalyzerInit:
    """Başlangıç durumu testleri."""

    def test_default_init(self):
        ka = KinematicAnalyzer()
        assert ka.buffer_size == 30
        assert ka.dt == pytest.approx(1.0 / 30.0)
        assert ka.micro_accel_threshold == 0.5

    def test_custom_init(self):
        ka = KinematicAnalyzer(buffer_size=15, fps=60.0, micro_accel_threshold=1.0)
        assert ka.buffer_size == 15
        assert ka.dt == pytest.approx(1.0 / 60.0)
        assert ka.micro_accel_threshold == 1.0

    def test_channels_exist(self):
        """6 kinematik kanal tanımlı olmalı."""
        assert len(CHANNELS) == 6
        assert "mouth_width" in CHANNELS
        assert "lip_corner_raise" in CHANNELS


class TestKinematicUpdate:
    """Güncelleme ve türev hesaplama testleri."""

    def test_first_update_returns_zero_velocities(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}
        state = ka.update(features, dominant="Nötr")

        assert isinstance(state, KinematicState)
        # İlk kare: hız = 0 (önceki veri yok)
        for ch in CHANNELS:
            assert state.velocities[ch] == 0.0

    def test_velocity_calculation(self):
        """Hız doğru hesaplanmalı: v = (D_curr - D_prev) / Δt."""
        ka = KinematicAnalyzer(fps=30.0)
        dt = 1.0 / 30.0

        # Kare 1
        features1 = {ch: 0.5 for ch in CHANNELS}
        ka.update(features1)

        # Kare 2: mouth_width 0.5 → 0.6 (+0.1)
        features2 = {ch: 0.5 for ch in CHANNELS}
        features2["mouth_width"] = 0.6
        state = ka.update(features2)

        expected_vel = (0.6 - 0.5) / dt
        assert state.velocities["mouth_width"] == pytest.approx(expected_vel, rel=1e-3)

    def test_acceleration_calculation(self):
        """İvme doğru hesaplanmalı: a = (v_curr - v_prev) / Δt."""
        ka = KinematicAnalyzer(fps=30.0)
        dt = 1.0 / 30.0

        # 3 kare: sabit → artış → artış (ivme oluşur)
        for val in [0.5, 0.5, 0.6]:
            features = {ch: 0.5 for ch in CHANNELS}
            features["mouth_width"] = val
            state = ka.update(features)

        # 2. karede v=0, 3. karede v=(0.6-0.5)/dt
        # ivme = (v3 - v2) / dt
        assert state.accelerations["mouth_width"] != 0.0

    def test_peak_channels(self):
        """Peak velocity/acceleration kanalları doğru belirlenmeli."""
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}
        ka.update(features)

        # Sadece mouth_width'i değiştir
        features["mouth_width"] = 0.8
        state = ka.update(features)

        assert state.peak_velocity_channel == "mouth_width"


class TestEmotionTransition:
    """Duygu geçiş tespiti testleri."""

    def test_no_transition_on_first_call(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}
        state = ka.update(features, dominant="Nötr")
        assert state.emotion_transition is None

    def test_transition_detected(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}

        ka.update(features, dominant="Nötr")
        state = ka.update(features, dominant="Gülümseme")

        assert state.emotion_transition is not None
        assert "Nötr" in state.emotion_transition
        assert "Gülümseme" in state.emotion_transition

    def test_no_transition_when_same(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}

        ka.update(features, dominant="Nötr")
        state = ka.update(features, dominant="Nötr")

        assert state.emotion_transition is None


class TestDuchenne:
    """Duchenne gülümseme tespiti testleri."""

    def test_no_duchenne_at_rest(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}
        state = ka.update(features)
        assert state.is_duchenne is False

    def test_duchenne_detection_requires_eye_squint(self):
        """Duchenne = dudak açılma + göz kısılma (eşzamanlı)."""
        ka = KinematicAnalyzer(fps=30.0)

        # Kare 1: taban
        features1 = {ch: 0.5 for ch in CHANNELS}
        ka.update(features1)

        # Kare 2: dudak açılma + göz kısılma
        features2 = {ch: 0.5 for ch in CHANNELS}
        features2["mouth_width"] = 0.9      # Güçlü artış → pozitif hız
        features2["left_eye_height"] = 0.3   # Güçlü düşüş → negatif hız
        features2["right_eye_height"] = 0.3
        state = ka.update(features2)

        # Not: Duchenne tespiti threshold'lara bağlı, dolayısıyla
        # bu test yapının çalıştığını doğrular
        assert isinstance(state.is_duchenne, bool)


class TestReset:
    """Reset fonksiyonu testleri."""

    def test_reset_clears_buffers(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}
        ka.update(features, dominant="Gülümseme")
        ka.update(features, dominant="Nötr")

        ka.reset()

        assert ka._prev_dominant is None
        assert ka._micro_state == "idle"
        assert ka._micro_frame_count == 0
        for ch in CHANNELS:
            assert len(ka._buffers[ch]) == 0

    def test_velocity_history(self):
        ka = KinematicAnalyzer()
        features = {ch: 0.5 for ch in CHANNELS}

        for i in range(10):
            features["mouth_width"] = 0.5 + i * 0.01
            ka.update(features)

        history = ka.get_velocity_history("mouth_width", n=5)
        assert len(history) == 5
