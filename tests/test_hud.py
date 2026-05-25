"""
tests/test_hud.py
─────────────────
HUDRenderer birim testleri.

Test edilen modül: ui/hud_renderer.py
"""

import pytest
import numpy as np
from unittest.mock import patch
from ui.hud_renderer import HUDRenderer, EMOTION_BGR, FATIGUE_BGR, TRACKING_BGR


class TestHUDRendererInit:
    """Başlangıç durumu testleri."""

    def test_default_init(self):
        hud = HUDRenderer()
        assert hud.panel_width == 180
        assert hud.temp_throttle_sec == 2.0

    def test_custom_init(self):
        hud = HUDRenderer(panel_width=200, temp_throttle_sec=5.0)
        assert hud.panel_width == 200
        assert hud.temp_throttle_sec == 5.0


class TestColorMappings:
    """Renk eşleme testleri."""

    def test_emotion_bgr_has_all_emotions(self):
        expected = {"Gülümseme", "Kaş Çatma", "Şaşırma", "Nötr"}
        assert set(EMOTION_BGR.keys()) == expected

    def test_fatigue_bgr_has_all_levels(self):
        expected = {"Optimal", "Normal", "Yorgun", "Tehlike"}
        assert set(FATIGUE_BGR.keys()) == expected

    def test_tracking_bgr_has_both_modes(self):
        assert True in TRACKING_BGR
        assert False in TRACKING_BGR

    def test_colors_are_bgr_tuples(self):
        """Tüm renkler 3 elemanlı (B, G, R) tuple olmalı."""
        for name, color in EMOTION_BGR.items():
            assert len(color) == 3, f"{name} rengi 3 elemanlı değil"
            assert all(0 <= c <= 255 for c in color), f"{name} renk değeri aralık dışı"


class TestCPUTemp:
    """CPU sıcaklık okuma testleri."""

    def test_cpu_temp_returns_float(self):
        hud = HUDRenderer()
        temp = hud.get_cpu_temp()
        assert isinstance(temp, float)

    def test_cpu_temp_fallback_on_pc(self):
        """PC'de termal dosya yoksa 0.0 dönmeli."""
        hud = HUDRenderer()
        hud._temp_available = False
        assert hud.get_cpu_temp() == 0.0

    def test_cpu_temp_throttling(self):
        """Throttle süresi içinde cache dönmeli."""
        hud = HUDRenderer(temp_throttle_sec=10.0)
        hud._last_temp = 55.0
        hud._last_temp_time = float("inf")  # Her zaman cache dön

        temp = hud.get_cpu_temp()
        assert temp == 55.0


class TestRender:
    """Ana render fonksiyonu testleri."""

    def _make_frame(self, width=640, height=480):
        """Test frame oluşturur."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_render_returns_same_shape(self):
        """render() aynı boyutta frame dönmeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        result = hud.render(frame)
        assert result.shape == frame.shape

    def test_render_returns_same_reference(self):
        """render() in-place çalışmalı (aynı referans)."""
        hud = HUDRenderer()
        frame = self._make_frame()
        result = hud.render(frame)
        assert result is frame

    def test_render_modifies_frame(self):
        """render() frame'e çizim yapmalı (boş frame'den farklı olmalı)."""
        hud = HUDRenderer()
        frame = self._make_frame()
        original_sum = frame.sum()

        hud.render(
            frame,
            fps=30,
            inference_latency=25.0,
            prediction_text="merhaba",
            prediction_conf=0.87,
        )

        assert frame.sum() > original_sum  # Bir şeyler çizildi

    def test_render_with_roi(self):
        """ROI bbox ile render hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        hud.render(frame, roi_bbox=(100, 200, 200, 300))

    def test_render_with_expressions(self):
        """Expression verisi ile render hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        expressions = {
            "dominant": "Gülümseme",
            "confidence": 0.85,
            "scores": {"Gülümseme": 0.85, "Kaş Çatma": 0.1, "Şaşırma": 0.05},
            "cognitive": {
                "ear": 0.31,
                "blink_rate": 18,
                "perclos": 0.05,
                "cognitive_load": 0.3,
                "fatigue_level": "Optimal",
            },
            "kinematic": {
                "micro_expression": None,
                "is_duchenne": True,
                "emotion_transition": None,
            },
        }
        hud.render(frame, expressions=expressions, mimic_mode=True)

    def test_render_with_landmarks(self):
        """Lip landmarks ile render hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        landmarks = [(200 + i * 5, 300 + (i % 3) * 3) for i in range(20)]
        hud.render(frame, lip_landmarks=landmarks)

    def test_render_all_tracking_modes(self):
        """Tüm tracking modları hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()

        hud.render(frame, tracking_mode="detection", tracking_quality=0.0)
        hud.render(frame, tracking_mode="tracking", tracking_quality=0.95)

    def test_render_turkish_prediction(self):
        """Türkçe karakter içeren tahmin hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        hud.render(
            frame,
            prediction_text="teşekkürler",
            prediction_conf=0.75,
        )

    def test_render_empty_prediction(self):
        """Boş tahmin hata vermemeli."""
        hud = HUDRenderer()
        frame = self._make_frame()
        hud.render(frame, prediction_text="", prediction_conf=0.0)


class TestRenderNoMimic:
    """Mimik analizi kapalı modda testler."""

    def test_render_without_mimic(self):
        """mimic_mode=False ile render çalışmalı."""
        hud = HUDRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = hud.render(frame, mimic_mode=False, fps=25)
        assert result is frame
