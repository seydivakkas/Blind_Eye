"""
tests/test_gpio.py
──────────────────
GPIOAlert birim testleri.

Test edilen modül: backend/gpio_alert.py
Mock modda (GPIO donanımı olmadan) test eder.
"""

import pytest
import logging
from backend.gpio_alert import GPIOAlert, THRESHOLD_LOW, THRESHOLD_MEDIUM, THRESHOLD_HIGH


class TestGPIOAlertInit:
    """Başlangıç durumu testleri."""

    def test_mock_init(self):
        """Mock modda GPIO'ya erişmeden başlatılabilmeli."""
        alert = GPIOAlert(mock=True)
        assert alert.mock is True
        assert alert._current_level == "low"

    def test_auto_fallback_to_mock(self):
        """PC'de RPi.GPIO bulunamazsa otomatik mock'a düşmeli."""
        alert = GPIOAlert(mock=False)  # PC'de RPi.GPIO yok
        assert alert.mock is True
        assert alert._available is False


class TestCognitiveLevel:
    """Bilişsel yük seviye testleri."""

    def test_low_level(self):
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(0.1)
        assert alert._current_level == "low"

    def test_medium_level(self):
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(0.45)
        assert alert._current_level == "medium"

    def test_high_level(self):
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(0.7)
        assert alert._current_level == "high"

    def test_critical_level(self):
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(0.9)
        assert alert._current_level == "critical"

    def test_clamping_above_1(self):
        """1.0 üstü değerler 1.0'a clamp edilmeli."""
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(1.5)
        assert alert._current_level == "critical"

    def test_clamping_below_0(self):
        """Negatif değerler 0.0'a clamp edilmeli."""
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(-0.5)
        assert alert._current_level == "low"

    def test_no_change_same_level(self):
        """Aynı seviye tekrar geldiğinde güncelleme yapılmamalı."""
        alert = GPIOAlert(mock=True)
        alert.set_cognitive_level(0.1)  # low
        alert.set_cognitive_level(0.15)  # hala low
        assert alert._current_level == "low"

    def test_level_transitions(self):
        """Seviye geçişleri doğru çalışmalı."""
        alert = GPIOAlert(mock=True)

        alert.set_cognitive_level(0.1)
        assert alert._current_level == "low"

        alert.set_cognitive_level(0.5)
        assert alert._current_level == "medium"

        alert.set_cognitive_level(0.7)
        assert alert._current_level == "high"

        alert.set_cognitive_level(0.9)
        assert alert._current_level == "critical"

        # Geri dönüş
        alert.set_cognitive_level(0.2)
        assert alert._current_level == "low"


class TestConfirmPrediction:
    """Tahmin onay testleri."""

    def test_confirm_does_not_raise(self):
        """Mock modda confirm_prediction hata vermemeli."""
        alert = GPIOAlert(mock=True)
        alert.confirm_prediction("merhaba")
        alert.confirm_prediction("teşekkürler")

    def test_confirm_turkish_words(self):
        """Türkçe kelimeler kabul edilmeli."""
        alert = GPIOAlert(mock=True)
        turkish_words = ["günaydın", "öğretmen", "çocuklar", "işaret"]
        for word in turkish_words:
            alert.confirm_prediction(word)  # Hata fırlatmamalı


class TestCleanup:
    """Temizleme testleri."""

    def test_cleanup_safe_in_mock(self):
        """Mock modda cleanup hata vermemeli."""
        alert = GPIOAlert(mock=True)
        alert.cleanup()  # Güvenli olmalı

    def test_double_cleanup(self):
        """İki kez cleanup çağırmak güvenli olmalı."""
        alert = GPIOAlert(mock=True)
        alert.cleanup()
        alert.cleanup()

    def test_destructor(self):
        """__del__ hata vermemeli."""
        alert = GPIOAlert(mock=True)
        del alert  # Hata fırlatmamalı


class TestThresholds:
    """Eşik değer doğruluk testleri."""

    def test_threshold_ordering(self):
        """Eşikler artan sırada olmalı."""
        assert THRESHOLD_LOW < THRESHOLD_MEDIUM < THRESHOLD_HIGH

    def test_threshold_values(self):
        """Eşik değerleri beklenen aralıkta olmalı."""
        assert 0.0 < THRESHOLD_LOW < 0.5
        assert 0.3 < THRESHOLD_MEDIUM < 0.8
        assert 0.6 < THRESHOLD_HIGH < 1.0
