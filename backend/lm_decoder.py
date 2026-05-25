"""
backend/lm_decoder.py
KenLM + pyctcdecode Beam Search Decoder

CTC logits + n-gram dil modeli ile WER'i %10-20 düşürür.
pyctcdecode yoksa otomatik olarak mevcut greedy decoder'a fallback yapar.

Kullanım:
    from backend.lm_decoder import LMDecoder
    decoder = LMDecoder(lm_path="models/tr_3gram.arpa")
    text, conf = decoder.decode(logits)
"""

import os
import json
import logging
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# pyctcdecode import
try:
    from pyctcdecode import build_ctcdecoder
    PYCTC_AVAILABLE = True
except ImportError:
    PYCTC_AVAILABLE = False
    logger.info("pyctcdecode yüklü değil — greedy decoder kullanılacak.")

# ── Vocab yükleme ──
_VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "vocab.json"
)


def _load_vocab() -> List[str]:
    """configs/vocab.json'dan karakter listesini yükler."""
    try:
        with open(_VOCAB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["charset"]
    except Exception:
        return list("<blank>") + list("abcçdefgğhıijklmnoöprsştuüvyz ")


class LMDecoder:
    """KenLM entegreli CTC Beam Search Decoder.

    Attributes:
        decoder: pyctcdecode BeamSearchDecoder instance
        alpha: Dil modeli ağırlığı (0.0-2.0)
        beta: Kelime ekleme bonusu
        beam_width: Beam genişliği
    """

    def __init__(
        self,
        lm_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        beam_width: int = 100,
        vocab: Optional[List[str]] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.vocab = vocab or _load_vocab()
        self.decoder = None
        self._use_beam = False

        if PYCTC_AVAILABLE:
            try:
                # pyctcdecode labels formatı: "" (blank) + gerçek karakterler
                labels = self._prepare_labels()

                if lm_path and os.path.exists(lm_path):
                    self.decoder = build_ctcdecoder(
                        labels=labels,
                        kenlm_model_path=lm_path,
                        alpha=alpha,
                        beta=beta,
                    )
                    self._use_beam = True
                    logger.info(
                        f"LM Decoder hazır: {lm_path} "
                        f"(α={alpha}, β={beta}, beam={beam_width})"
                    )
                else:
                    # LM dosyası yok — beam search sadece (LM'siz)
                    self.decoder = build_ctcdecoder(labels=labels)
                    self._use_beam = True
                    logger.info("Beam Search Decoder hazır (LM yok, sadece beam search)")

            except Exception as e:
                logger.warning(f"pyctcdecode başlatılamadı: {e}")
                self._use_beam = False

        if not self._use_beam:
            logger.info("Greedy CTC Decoder aktif (fallback)")

    def _prepare_labels(self) -> List[str]:
        """pyctcdecode için label listesi hazırlar.

        pyctcdecode blank="" olarak bekler, charset[0]='<blank>'
        """
        labels = []
        for token in self.vocab:
            if token == "<blank>":
                labels.append("")  # pyctcdecode blank formatı
            else:
                labels.append(token)
        return labels

    def decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """CTC logits → metin çözümleme.

        Args:
            logits: [B, T, V] veya [T, V] boyutlu model çıktısı

        Returns:
            (text, confidence) tuple'ı
        """
        if logits is None:
            return "", 0.0

        # Squeeze batch dimension
        if logits.ndim == 3:
            logits = logits.squeeze(0)  # [T, V]

        if self._use_beam:
            return self._beam_decode(logits)
        return self._greedy_decode(logits)

    def _beam_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """pyctcdecode beam search."""
        try:
            text = self.decoder.decode(
                logits,
                beam_width=self.beam_width,
            )
            # Confidence hesaplama
            probs = self._softmax(logits)
            conf = float(np.mean(np.max(probs, axis=-1)))
            return text.strip(), max(min(conf, 1.0), 0.0)

        except Exception as e:
            logger.warning(f"Beam decode hatası, greedy fallback: {e}")
            return self._greedy_decode(logits)

    def _greedy_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """Greedy CTC decode (fallback)."""
        probs = self._softmax(logits)
        tokens = np.argmax(probs, axis=-1)

        # CTC collapse
        cleaned = []
        blank_idx = 0  # <blank> = index 0
        for t in tokens:
            if t != blank_idx and (not cleaned or t != cleaned[-1]):
                cleaned.append(int(t))

        text = "".join(
            self.vocab[i] if i < len(self.vocab) else " "
            for i in cleaned
        )

        conf = float(np.mean(np.max(probs, axis=-1)))
        return text.strip(), max(min(conf, 1.0), 0.0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
