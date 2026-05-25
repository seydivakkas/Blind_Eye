"""
tests/test_decoder.py — TurkishCTCDecoder birim testleri.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.decoder import TurkishCTCDecoder


@pytest.fixture
def decoder(turkish_vocab):
    return TurkishCTCDecoder(vocab=turkish_vocab, blank_idx=0)


class TestCTCGreedyDecode:
    def test_basic_decode(self, decoder):
        """Basit bir token dizisini decode eder."""
        logits = np.zeros((1, 5, 30))
        logits[0, 0, 1] = 10.0   # 'a'
        logits[0, 1, 2] = 10.0   # 'b'
        logits[0, 2, 0] = 10.0   # blank
        logits[0, 3, 3] = 10.0   # 'c'
        logits[0, 4, 0] = 10.0   # blank

        text, conf = decoder.decode(logits)
        assert len(text) > 0
        assert 0.0 <= conf <= 1.0

    def test_none_logits_returns_empty(self, decoder):
        """None logits → boş string."""
        text, conf = decoder.decode(None)
        assert text == ""
        assert conf == 0.0

    def test_confidence_range(self, decoder, dummy_logits):
        """Confidence 0-1 aralığında olmalı."""
        text, conf = decoder.decode(dummy_logits)
        assert 0.0 <= conf <= 1.0

    def test_repeated_tokens_cleaned(self, decoder):
        """Tekrarlanan tokenler temizlenmeli (CTC collapse)."""
        logits = np.zeros((1, 4, 30))
        logits[0, 0, 1] = 10.0   # 'a'
        logits[0, 1, 1] = 10.0   # 'a' (tekrar — CTC collapse)
        logits[0, 2, 1] = 10.0   # 'a' (tekrar — CTC collapse)
        logits[0, 3, 2] = 10.0   # 'b'

        text, conf = decoder.decode(logits)
        # CTC greedy: ardışık aynı tokenler birleştirilir → "ab"
        assert "aaa" not in text


class TestRegexCleaning:
    def test_special_chars_removed(self, decoder):
        """Özel karakterler temizlenmeli."""
        logits = np.zeros((1, 3, 30))
        logits[0, 0, 1] = 10.0  # 'a'
        logits[0, 1, 2] = 10.0  # 'b'
        logits[0, 2, 3] = 10.0  # 'c'

        text, conf = decoder.decode(logits)
        # Sadece alfanümerik ve izin verilen karakter olmalı
        assert all(c.isalnum() or c.isspace() or c in ".,!?-'" for c in text)
