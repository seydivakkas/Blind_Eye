"""
tests/conftest.py — Paylaşılan fixture'lar.
"""
import pytest
import numpy as np


@pytest.fixture
def dummy_roi_chunk():
    """[T=6, H=96, W=96, C=1] boyutunda rastgele ROI chunk'ı."""
    return np.random.rand(6, 96, 96, 1).astype(np.float32)


@pytest.fixture
def dummy_logits():
    """[1, T=6, V=30] boyutunda rastgele logits."""
    return np.random.randn(1, 6, 30).astype(np.float32)


@pytest.fixture
def turkish_vocab():
    """Türkçe karakter listesi (blank=0)."""
    return list(" abcçdefgğhıijklmnoöprsştuüvyz")
