"""
backend/viseme_decoder.py
─────────────────────────
Viseme dizisinden kelime eşleme decoder'ı.

CTC çıktısından gelen viseme sınıf dizisini, 16 kelimelik sözlük
üzerinde Levenshtein mesafesiyle en yakın kelimeye eşler.

Kullanım:
    from backend.viseme_decoder import VisemeDecoder
    
    decoder = VisemeDecoder("configs/tr_viseme_map.json", "configs/viseme_vocab.json")
    word, confidence = decoder.decode([2, 5, 3, 7, 5, 2, 5])  # viseme indisleri
"""

import os
import json
import logging
import numpy as np
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)


def _levenshtein(a: list, b: list) -> int:
    """İki liste arasındaki Levenshtein düzenleme mesafesini hesaplar."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


class VisemeDecoder:
    """Viseme CTC çıktısından kelime eşleme.

    İki modda çalışır:
    1. Sözlük tabanlı (dictionary-based): En yakın kelime eşlemesi
    2. Ham viseme dizisi çıktısı (debug/analiz için)
    """

    def __init__(
        self,
        viseme_map_path: str = "configs/tr_viseme_map.json",
        viseme_vocab_path: str = "configs/viseme_vocab.json",
        word_labels_path: str = "data/processed/labels.json",
    ):
        # Viseme vocab yükle
        with open(viseme_vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.viseme_classes = vocab_data["charset"]  # ['<blank>', 'V_ALVEOLAR_FRICATIVE', ...]
        self.num_classes = vocab_data["num_classes"]
        self.blank_idx = vocab_data["blank_idx"]

        # Fonem→viseme haritası
        with open(viseme_map_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)
        self.phoneme_to_viseme = map_data["phoneme_to_viseme"]

        # Sözlük oluştur: kelime → viseme indis dizisi
        self.word_dict: Dict[str, List[int]] = {}
        self._build_dictionary(word_labels_path)

        logger.info(
            f"VisemeDecoder hazır: {self.num_classes} sınıf, "
            f"{len(self.word_dict)} kelime sözlüğü"
        )

    def _build_dictionary(self, labels_path: str):
        """labels.json'dan benzersiz kelimeleri alır, her birinin viseme indis dizisini oluşturur."""
        if not os.path.exists(labels_path):
            logger.warning(f"labels.json bulunamadı: {labels_path}")
            return

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        unique_words = set(labels.values())
        viseme_to_idx = {v: i for i, v in enumerate(self.viseme_classes)}

        for word in unique_words:
            viseme_seq = []
            for char in word.lower():
                viseme_name = self.phoneme_to_viseme.get(char)
                if viseme_name and viseme_name != "V_SILENCE":
                    idx = viseme_to_idx.get(viseme_name)
                    if idx is not None:
                        viseme_seq.append(idx)
            if viseme_seq:
                self.word_dict[word] = viseme_seq

    def decode(self, viseme_ids: List[int]) -> Tuple[str, float]:
        """Viseme indis dizisinden kelime eşleme.

        Args:
            viseme_ids: CTC greedy/beam decode sonrası viseme indisleri (blank ve tekrarlar zaten çıkarılmış)

        Returns:
            (kelime, güven_skoru) tuple'ı
        """
        if not viseme_ids or not self.word_dict:
            return "", 0.0

        best_word = ""
        best_distance = float("inf")
        best_max_len = 1

        for word, ref_ids in self.word_dict.items():
            dist = _levenshtein(viseme_ids, ref_ids)
            max_len = max(len(viseme_ids), len(ref_ids))

            if dist < best_distance or (dist == best_distance and max_len < best_max_len):
                best_distance = dist
                best_word = word
                best_max_len = max_len

        # Güven skoru: 1 - (distance / max_len)
        confidence = max(0.0, 1.0 - (best_distance / max(best_max_len, 1)))
        return best_word, round(confidence, 4)

    def decode_logits(self, logits: np.ndarray) -> Tuple[str, float]:
        """Ham CTC logits'ten kelime çözümleme.

        Args:
            logits: [T, num_viseme_classes] boyutlu model çıktısı

        Returns:
            (kelime, güven_skoru) tuple'ı
        """
        if logits is None or logits.size == 0:
            return "", 0.0

        # Squeeze batch dimension
        if logits.ndim == 3:
            logits = logits.squeeze(0)

        # Greedy CTC decode
        token_ids = np.argmax(logits, axis=-1).tolist()

        # CTC collapse: blank ve ardışık tekrarları kaldır
        collapsed = []
        prev = None
        for t in token_ids:
            if t != self.blank_idx and t != prev:
                collapsed.append(t)
            prev = t

        return self.decode(collapsed)

    def decode_raw(self, viseme_ids: List[int]) -> str:
        """Viseme indislerini okunabilir isimlere çevirir (debug için)."""
        names = []
        for idx in viseme_ids:
            if 0 <= idx < len(self.viseme_classes):
                names.append(self.viseme_classes[idx])
            else:
                names.append(f"?{idx}")
        return " → ".join(names)

    def get_word_viseme_table(self) -> Dict[str, str]:
        """Sözlükteki tüm kelimelerin viseme dizilerini döndürür."""
        table = {}
        for word, ids in self.word_dict.items():
            viseme_names = [self.viseme_classes[i] for i in ids]
            table[word] = " ".join(viseme_names)
        return table
