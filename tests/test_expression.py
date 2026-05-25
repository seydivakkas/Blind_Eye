"""
tests/test_expression.py — ExpressionDetector birim testleri.

3 katmanlı analiz çıktısını test eder:
    Katman 1: Geometrik mimik skorları (statik)
    Katman 2: Kinematik türev analizi (zamansal)
    Katman 3: Bilişsel yük ve yorgunluk (EAR/PERCLOS)
"""
import pytest
import sys
import os
from collections import namedtuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.expression_detector import ExpressionDetector

Point = namedtuple("Point", ["x", "y"])


# Landmark'lar .z attribute'u olmadan da çalışmalı (Point ile test)
# Ancak CognitiveMonitor .x ve .y bekler → uyumlu mock
class MockLandmark:
    """MediaPipe uyumlu mock landmark (x, y, z)."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _create_full_landmarks(n=468, x=0.5, y=0.5):
    """468 MockLandmark oluşturur."""
    return [MockLandmark(x, y) for _ in range(n)]


@pytest.fixture
def detector():
    return ExpressionDetector()


class TestEmptyInput:
    """Boş ve geçersiz giriş testleri."""

    def test_empty_landmarks(self, detector):
        """Boş nirengi noktaları listesi Nötr dönmeli."""
        res = detector.detect([])
        assert res["dominant"] == "Nötr"
        assert res["confidence"] == 1.0
        assert "scores" in res
        assert res["scores"]["Gülümseme"] == 0.0

    def test_empty_has_kinematic(self, detector):
        """Boş giriş kinematik alanı içermeli."""
        res = detector.detect([])
        assert "kinematic" in res
        assert res["kinematic"]["micro_expression"] is None
        assert res["kinematic"]["is_duchenne"] is False

    def test_empty_has_cognitive(self, detector):
        """Boş giriş bilişsel durum alanı içermeli."""
        res = detector.detect([])
        assert "cognitive" in res
        assert res["cognitive"]["fatigue_level"] == "Optimal"


class TestNeutralExpression:
    """Nötr yüz ifadesi testleri."""

    def test_neutral_landmarks(self, detector):
        """Nötr yüz ifadesini temsil eden nirengi noktaları testi."""
        landmarks = _create_full_landmarks()

        # Göz dış kenarları (normalizasyon skalası)
        landmarks[33] = MockLandmark(0.4, 0.4)
        landmarks[263] = MockLandmark(0.6, 0.4)

        # Nötr kaşlar
        landmarks[107] = MockLandmark(0.45, 0.3)
        landmarks[336] = MockLandmark(0.55, 0.3)

        # Nötr ağız
        landmarks[61] = MockLandmark(0.43, 0.6)
        landmarks[291] = MockLandmark(0.57, 0.6)
        landmarks[0] = MockLandmark(0.5, 0.59)
        landmarks[17] = MockLandmark(0.5, 0.61)

        # İç dudak noktaları (şaşırma için)
        landmarks[13] = MockLandmark(0.5, 0.595)
        landmarks[14] = MockLandmark(0.5, 0.605)

        # Göz dikey noktaları
        landmarks[159] = MockLandmark(0.42, 0.385)
        landmarks[145] = MockLandmark(0.42, 0.415)
        landmarks[386] = MockLandmark(0.58, 0.385)
        landmarks[374] = MockLandmark(0.58, 0.415)

        # Sol göz EAR noktaları
        landmarks[160] = MockLandmark(0.40, 0.37)
        landmarks[158] = MockLandmark(0.40, 0.37)
        landmarks[133] = MockLandmark(0.45, 0.4)
        landmarks[153] = MockLandmark(0.40, 0.43)
        landmarks[144] = MockLandmark(0.40, 0.43)

        # Sağ göz EAR noktaları
        landmarks[362] = MockLandmark(0.55, 0.4)
        landmarks[385] = MockLandmark(0.60, 0.37)
        landmarks[387] = MockLandmark(0.60, 0.37)
        landmarks[373] = MockLandmark(0.60, 0.43)
        landmarks[380] = MockLandmark(0.60, 0.43)

        res = detector.detect(landmarks)
        assert res["dominant"] == "Nötr"
        assert res["scores"]["Gülümseme"] < 0.35
        assert res["scores"]["Kaş Çatma"] < 0.35
        assert res["scores"]["Şaşırma"] < 0.35


class TestSmileDetection:
    """Gülümseme tespiti testleri."""

    def test_smile(self, detector):
        """Geniş ve yukarı kalkmış ağız köşeleri gülümseme olarak tespiti."""
        landmarks = _create_full_landmarks()

        landmarks[33] = MockLandmark(0.4, 0.4)
        landmarks[263] = MockLandmark(0.6, 0.4)
        landmarks[107] = MockLandmark(0.45, 0.3)
        landmarks[336] = MockLandmark(0.55, 0.3)

        # Geniş ağız
        landmarks[61] = MockLandmark(0.41, 0.58)
        landmarks[291] = MockLandmark(0.59, 0.58)
        landmarks[0] = MockLandmark(0.5, 0.59)
        landmarks[17] = MockLandmark(0.5, 0.63)

        landmarks[13] = MockLandmark(0.5, 0.595)
        landmarks[14] = MockLandmark(0.5, 0.625)

        landmarks[159] = MockLandmark(0.42, 0.385)
        landmarks[145] = MockLandmark(0.42, 0.415)
        landmarks[386] = MockLandmark(0.58, 0.385)
        landmarks[374] = MockLandmark(0.58, 0.415)

        # Göz EAR noktaları
        landmarks[160] = MockLandmark(0.40, 0.37)
        landmarks[158] = MockLandmark(0.40, 0.37)
        landmarks[133] = MockLandmark(0.45, 0.4)
        landmarks[153] = MockLandmark(0.40, 0.43)
        landmarks[144] = MockLandmark(0.40, 0.43)
        landmarks[362] = MockLandmark(0.55, 0.4)
        landmarks[385] = MockLandmark(0.60, 0.37)
        landmarks[387] = MockLandmark(0.60, 0.37)
        landmarks[373] = MockLandmark(0.60, 0.43)
        landmarks[380] = MockLandmark(0.60, 0.43)

        res = detector.detect(landmarks)
        assert res["scores"]["Gülümseme"] > 0.6
        assert res["dominant"] == "Gülümseme"


class TestFrownDetection:
    """Kaş çatma tespiti testleri."""

    def test_frown(self, detector):
        """İç kaşların yakınlaştığı kaş çatma tespiti."""
        landmarks = _create_full_landmarks()

        landmarks[33] = MockLandmark(0.4, 0.4)
        landmarks[263] = MockLandmark(0.6, 0.4)

        # Çatık kaşlar
        landmarks[107] = MockLandmark(0.465, 0.3)
        landmarks[336] = MockLandmark(0.535, 0.3)

        landmarks[61] = MockLandmark(0.43, 0.6)
        landmarks[291] = MockLandmark(0.57, 0.6)
        landmarks[0] = MockLandmark(0.5, 0.59)
        landmarks[17] = MockLandmark(0.5, 0.61)

        landmarks[13] = MockLandmark(0.5, 0.595)
        landmarks[14] = MockLandmark(0.5, 0.605)

        landmarks[159] = MockLandmark(0.42, 0.385)
        landmarks[145] = MockLandmark(0.42, 0.415)
        landmarks[386] = MockLandmark(0.58, 0.385)
        landmarks[374] = MockLandmark(0.58, 0.415)

        landmarks[160] = MockLandmark(0.40, 0.37)
        landmarks[158] = MockLandmark(0.40, 0.37)
        landmarks[133] = MockLandmark(0.45, 0.4)
        landmarks[153] = MockLandmark(0.40, 0.43)
        landmarks[144] = MockLandmark(0.40, 0.43)
        landmarks[362] = MockLandmark(0.55, 0.4)
        landmarks[385] = MockLandmark(0.60, 0.37)
        landmarks[387] = MockLandmark(0.60, 0.37)
        landmarks[373] = MockLandmark(0.60, 0.43)
        landmarks[380] = MockLandmark(0.60, 0.43)

        res = detector.detect(landmarks)
        assert res["scores"]["Kaş Çatma"] > 0.7
        assert res["dominant"] == "Kaş Çatma"


class TestKinematicOutput:
    """Kinematik çıktı formatı testleri."""

    def test_kinematic_output_format(self, detector):
        """Kinematik alanları dönen dict'te var mı."""
        landmarks = _create_full_landmarks()
        landmarks[33] = MockLandmark(0.4, 0.4)
        landmarks[263] = MockLandmark(0.6, 0.4)
        landmarks[107] = MockLandmark(0.45, 0.3)
        landmarks[336] = MockLandmark(0.55, 0.3)
        landmarks[61] = MockLandmark(0.43, 0.6)
        landmarks[291] = MockLandmark(0.57, 0.6)
        landmarks[0] = MockLandmark(0.5, 0.59)
        landmarks[17] = MockLandmark(0.5, 0.61)
        landmarks[13] = MockLandmark(0.5, 0.595)
        landmarks[14] = MockLandmark(0.5, 0.605)
        landmarks[159] = MockLandmark(0.42, 0.385)
        landmarks[145] = MockLandmark(0.42, 0.415)
        landmarks[386] = MockLandmark(0.58, 0.385)
        landmarks[374] = MockLandmark(0.58, 0.415)
        landmarks[160] = MockLandmark(0.40, 0.37)
        landmarks[158] = MockLandmark(0.40, 0.37)
        landmarks[133] = MockLandmark(0.45, 0.4)
        landmarks[153] = MockLandmark(0.40, 0.43)
        landmarks[144] = MockLandmark(0.40, 0.43)
        landmarks[362] = MockLandmark(0.55, 0.4)
        landmarks[385] = MockLandmark(0.60, 0.37)
        landmarks[387] = MockLandmark(0.60, 0.37)
        landmarks[373] = MockLandmark(0.60, 0.43)
        landmarks[380] = MockLandmark(0.60, 0.43)

        res = detector.detect(landmarks)

        assert "kinematic" in res
        kin = res["kinematic"]
        assert "velocities" in kin
        assert "accelerations" in kin
        assert "micro_expression" in kin
        assert "is_duchenne" in kin
        assert "emotion_transition" in kin

    def test_cognitive_output_format(self, detector):
        """Bilişsel durum alanları doğru formatta mı."""
        landmarks = _create_full_landmarks()
        landmarks[33] = MockLandmark(0.4, 0.4)
        landmarks[263] = MockLandmark(0.6, 0.4)
        landmarks[107] = MockLandmark(0.45, 0.3)
        landmarks[336] = MockLandmark(0.55, 0.3)
        landmarks[61] = MockLandmark(0.43, 0.6)
        landmarks[291] = MockLandmark(0.57, 0.6)
        landmarks[0] = MockLandmark(0.5, 0.59)
        landmarks[17] = MockLandmark(0.5, 0.61)
        landmarks[13] = MockLandmark(0.5, 0.595)
        landmarks[14] = MockLandmark(0.5, 0.605)
        landmarks[159] = MockLandmark(0.42, 0.385)
        landmarks[145] = MockLandmark(0.42, 0.415)
        landmarks[386] = MockLandmark(0.58, 0.385)
        landmarks[374] = MockLandmark(0.58, 0.415)
        landmarks[160] = MockLandmark(0.40, 0.37)
        landmarks[158] = MockLandmark(0.40, 0.37)
        landmarks[133] = MockLandmark(0.45, 0.4)
        landmarks[153] = MockLandmark(0.40, 0.43)
        landmarks[144] = MockLandmark(0.40, 0.43)
        landmarks[362] = MockLandmark(0.55, 0.4)
        landmarks[385] = MockLandmark(0.60, 0.37)
        landmarks[387] = MockLandmark(0.60, 0.37)
        landmarks[373] = MockLandmark(0.60, 0.43)
        landmarks[380] = MockLandmark(0.60, 0.43)

        res = detector.detect(landmarks)

        assert "cognitive" in res
        cog = res["cognitive"]
        assert "ear" in cog
        assert "blink_rate" in cog
        assert "perclos" in cog
        assert "cognitive_load" in cog
        assert "fatigue_level" in cog
        assert 0.0 <= cog["cognitive_load"] <= 1.0
        assert cog["fatigue_level"] in ("Optimal", "Normal", "Yorgun", "Tehlike")
