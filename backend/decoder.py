import numpy as np
import re
import json
import os
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# ── Vocab yükleme yardımcısı ──
_VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "vocab.json"
)


def load_vocab(path: str = _VOCAB_PATH) -> Tuple[List[str], int, int]:
    """
    configs/vocab.json'dan karakter setini yükler.

    Returns:
        (charset, blank_idx, num_classes)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        charset = data["charset"]
        blank_idx = data.get("blank_idx", 0)
        num_classes = data.get("num_classes", len(charset))
        logger.info(f"Vocab yüklendi: {num_classes} sınıf, blank_idx={blank_idx}")
        return charset, blank_idx, num_classes
    except FileNotFoundError:
        logger.warning(f"vocab.json bulunamadı ({path}), varsayılan kullanılıyor.")
        return None, 0, 30
    except Exception as e:
        logger.warning(f"vocab.json okunamadı: {e}, varsayılan kullanılıyor.")
        return None, 0, 30


class TurkishCTCDecoder:
    """CTC greedy decode + Türkçe regex temizleme.

    Karakter seti configs/vocab.json'dan yüklenir.
    Dosya bulunamazsa dahili TURKISH_VOCAB kullanılır.
    """

    TURKISH_VOCAB = list("<blank>") + list("abcçdefgğhıijklmnoöprsştuüvyz ")

    def __init__(self, vocab: Optional[List[str]] = None, blank_idx: int = 0):
        # Önce vocab.json'dan yüklemeyi dene
        if vocab is None:
            loaded_vocab, loaded_blank, _ = load_vocab()
            if loaded_vocab is not None:
                self.vocab = loaded_vocab
                self.blank = loaded_blank
            else:
                self.vocab = self.TURKISH_VOCAB
                self.blank = blank_idx
        else:
            self.vocab = vocab
            self.blank = blank_idx

        self.patterns = [
            (re.compile(r"(.)\1{2,}"), r"\1"),       # 3+ tekrar → 1
            (re.compile(r"[^\w\s.,!?'\-]"), ""),      # Özel karakter temizle
        ]

        logger.debug(
            f"Decoder başlatıldı: vocab_size={len(self.vocab)}, blank_idx={self.blank}"
        )

    def decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """CTC greedy decode.

        Args:
            logits: [B, T, V] veya [T, V] boyutlu model çıktısı

        Returns:
            (text, confidence) tuple'ı
        """
        if logits is None:
            return "", 0.0

        probs = self._softmax(logits)
        tokens = np.argmax(probs, axis=-1).squeeze()

        # CTC collapse: ardışık aynı token + blank kaldırma
        cleaned = []
        for t in tokens:
            if t != self.blank and (not cleaned or t != cleaned[-1]):
                cleaned.append(int(t))

        # Token → karakter dönüşümü
        text = "".join(
            self.vocab[i] if i < len(self.vocab) else " "
            for i in cleaned
        )

        # Regex temizleme
        for pattern, repl in self.patterns:
            text = pattern.sub(repl, text)

        conf = float(np.mean(np.max(probs, axis=-1)))
        return text.strip(), max(min(conf, 1.0), 0.0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
