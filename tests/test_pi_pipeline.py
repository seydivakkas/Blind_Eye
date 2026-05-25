"""
tests/test_pi_pipeline.py — Raspberry Pi 3 B+ Türkçe Dudak Okuma Pipeline birim testleri.
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pi_run import PiPipeline

MODEL_PATH = "models/pi_model_int8.onnx"

@pytest.fixture
def pi_pipeline():
    """Raspberry Pi 3 B+ Pipeline instance'ı oluşturur (INT8 ONNX modelini yükler)."""
    # Eğer test çalıştırılırken model yoksa testi atla
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Test için gerekli model bulunamadı: {MODEL_PATH}")
    return PiPipeline(model_path=MODEL_PATH, seq_len=6, roi_size=96)

class TestPiPipeline:
    def test_pipeline_initialization(self, pi_pipeline):
        """Pipeline'ın doğru şekilde yüklendiğini doğrular."""
        assert pi_pipeline.model_path == MODEL_PATH
        assert pi_pipeline.seq_len == 6
        assert pi_pipeline.roi_size == 96
        assert len(pi_pipeline.frame_buffer) == 0
        assert pi_pipeline.session is not None
        assert pi_pipeline.input_name is not None

    def test_preprocess_frame(self, pi_pipeline):
        """Preprocess_frame fonksiyonunun [96, 96] boyutlu, normalize edilmiş çıktı ürettiğini test eder."""
        # 640x480 RGB sahte kamera karesi oluştur
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Ekranın ortasından ROI alanını belirle
        roi_w, roi_h = 120, 90
        roi_x = (640 - roi_w) // 2
        roi_y = (480 - roi_h) // 2
        roi_coords = (roi_x, roi_y, roi_w, roi_h)
        
        # Ön işlem uygula
        processed = pi_pipeline.preprocess_frame(dummy_frame, roi_coords)
        
        # Boyut ve tip testleri
        assert processed.shape == (96, 96)
        assert processed.dtype == np.float32
        
        # Değerlerin 0..1 arasında olduğunu test et
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0

    def test_rolling_buffer_capacity(self, pi_pipeline):
        """Kayar pencere buffer limitlerinin (seq_len) doğru yönetildiğini test eder."""
        # 8 adet sahte kare ekle (seq_len = 6 olduğundan buffer 6 eleman tutmalı)
        for _ in range(8):
            dummy_crop = np.random.rand(96, 96).astype(np.float32)
            pi_pipeline.add_frame(dummy_crop)
            
        assert len(pi_pipeline.frame_buffer) == 6

    def test_clear_buffer(self, pi_pipeline):
        """Sıfırlama fonksiyonunun tampon belleği boşalttığını test eder."""
        # Kareler ekle
        for _ in range(4):
            dummy_crop = np.random.rand(96, 96).astype(np.float32)
            pi_pipeline.add_frame(dummy_crop)
            
        assert len(pi_pipeline.frame_buffer) == 4
        
        # Sıfırla
        pi_pipeline.clear_buffer()
        assert len(pi_pipeline.frame_buffer) == 0

    def test_prediction_pipeline(self, pi_pipeline):
        """Dizi dolduğunda tahmin ve CTC çözümlemesinin yapıldığını doğrular."""
        # seq_len kadar sahte kare ekle
        for _ in range(6):
            dummy_crop = np.random.rand(96, 96).astype(np.float32)
            pi_pipeline.add_frame(dummy_crop)
            
        # Tahmin yürüt
        text, confidence, latency = pi_pipeline.predict()
        
        # Çıktı tiplerini doğrula
        assert isinstance(text, str)
        assert isinstance(confidence, float)
        assert isinstance(latency, float)
        assert 0.0 <= confidence <= 1.0
        assert latency > 0.0
