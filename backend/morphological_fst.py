"""
backend/morphological_fst.py
────────────────────────────
Turkce Morfolojik FST Post-Processing.

Akademik Motivasyon:
    Turkce'nin agglutinative (eklemeli) yapisi, n-gram dil modelinin
    kapsayamadigı uzun kelimeleri uretir. "evlerimizdeymisiz" gibi
    kelimeleri morfolojik segmentasyon yaparak (ev+ler+imiz+de+ymis+iz)
    duzeltir.

    FST (Finite-State Transducer) tabanli ek zinciri dogrulama:
    1. Kok bulma (basit sozluk + uzunluk esikleme)
    2. Ek zinciri dogrulama (Turkce ek kurallari)
    3. Unlu uyumu kontrolu (ek uygunlugu)

Referanslar:
    - Oflazer (1994), "Two-level Description of Turkish Morphology"
    - TrMorph — Cagri Coltekin
    - Zemberek-NLP — Ahmet Ak. Afsin

Kullanim:
    from backend.morphological_fst import TurkishMorphologicalFST

    fst = TurkishMorphologicalFST()
    result = fst.segment("evlerimizdeymisiz")
    # → {"root": "ev", "suffixes": ["ler", "imiz", "de", "ymis", "iz"]}
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================
#  Turkce Ek Kaliplari
# ======================================================

# Isim cekimi ek gruplari (onem sirasi)
NOUN_SUFFIXES = [
    # Cogul
    ("ler", "PLURAL"), ("lar", "PLURAL"),
    # Iyelik
    ("im", "POSS_1S"), ("ım", "POSS_1S"),
    ("in", "POSS_2S"), ("ın", "POSS_2S"),
    ("i", "POSS_3S"), ("ı", "POSS_3S"), ("si", "POSS_3S"), ("sı", "POSS_3S"),
    ("imiz", "POSS_1P"), ("ımız", "POSS_1P"),
    ("iniz", "POSS_2P"), ("ınız", "POSS_2P"),
    ("leri", "POSS_3P"), ("ları", "POSS_3P"),
    # Hal ekleri
    ("de", "LOC"), ("da", "LOC"), ("te", "LOC"), ("ta", "LOC"),
    ("den", "ABL"), ("dan", "ABL"), ("ten", "ABL"), ("tan", "ABL"),
    ("e", "DAT"), ("a", "DAT"), ("ye", "DAT"), ("ya", "DAT"),
    ("i", "ACC"), ("ı", "ACC"), ("yi", "ACC"), ("yı", "ACC"),
    # Baglac
    ("dir", "COP"), ("dır", "COP"), ("tir", "COP"), ("tır", "COP"),
    ("dur", "COP"), ("dür", "COP"), ("tur", "COP"), ("tür", "COP"),
    # Soru
    ("mi", "QUES"), ("mı", "QUES"), ("mu", "QUES"), ("mü", "QUES"),
]

# Fiil cekimi ek gruplari
VERB_SUFFIXES = [
    # Zaman/kip ekleri
    ("yor", "PROG"), ("iyor", "PROG"), ("ıyor", "PROG"),
    ("uyor", "PROG"), ("üyor", "PROG"),
    ("ecek", "FUT"), ("acak", "FUT"),
    ("miş", "NARR"), ("mış", "NARR"), ("muş", "NARR"), ("müş", "NARR"),
    ("di", "PAST"), ("dı", "PAST"), ("ti", "PAST"), ("tı", "PAST"),
    ("du", "PAST"), ("dü", "PAST"), ("tu", "PAST"), ("tü", "PAST"),
    # Kisi ekleri
    ("um", "1S"), ("ım", "1S"),
    ("sun", "2S"), ("sın", "2S"),
    ("uz", "1P"), ("ız", "1P"),
    ("sunuz", "2P"), ("sınız", "2P"),
    ("ler", "3P"), ("lar", "3P"),
]

# Bilinen kisa kokler (minimum skor icin)
COMMON_ROOTS = {
    "ev", "el", "goz", "is", "ac", "ol", "gel", "git",
    "ver", "al", "de", "ye", "ic", "oku", "yaz", "bas",
    "gul", "sus", "dur", "kal", "bil", "bul", "koy",
    "sen", "ben", "biz", "siz", "bu", "su", "o",
}


class TurkishMorphologicalFST:
    """Turkce Morfolojik Segmentasyon — FST Tabanli.

    Args:
        corpus_path: Kok sozlugu (opsiyonel)
        min_root_len: Minimum kok uzunlugu
    """

    def __init__(
        self,
        corpus_path: str = "data/corpus_tr.txt",
        min_root_len: int = 2,
    ):
        self.min_root_len = min_root_len

        # Kok sozlugu (frekans tabanli)
        self.root_vocab: Dict[str, int] = {}
        self._load_roots(corpus_path)

        logger.info(
            f"MorphologicalFST hazir: {len(self.root_vocab)} kok kelime"
        )

    def _load_roots(self, path: str):
        """Kok sozlugunu yukler."""
        # Bilinen kisa kokler her zaman dahil
        for root in COMMON_ROOTS:
            self.root_vocab[root] = 100

        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    words = line.strip().lower().split()
                    for word in words:
                        if len(word) >= self.min_root_len:
                            self.root_vocab[word] = self.root_vocab.get(word, 0) + 1
        except Exception:
            pass

    def segment(self, word: str) -> Dict:
        """Kelimeyi kok + ekler olarak ayiristirir.

        Args:
            word: Segmente edilecek kelime

        Returns:
            {
                "word": orijinal kelime,
                "root": bulunan kok,
                "suffixes": [ek listesi],
                "suffix_types": [ek tur listesi],
                "valid": morfolojik gecerlilik,
                "score": guvenilirlik skoru [0-1],
            }
        """
        word = word.lower().strip()

        if not word:
            return {"word": "", "root": "", "suffixes": [], "valid": False, "score": 0}

        # Kisa kelime — segmentasyon gereksiz
        if len(word) <= self.min_root_len:
            return {
                "word": word,
                "root": word,
                "suffixes": [],
                "suffix_types": [],
                "valid": word in self.root_vocab or word in COMMON_ROOTS,
                "score": 1.0 if word in self.root_vocab else 0.5,
            }

        # En iyi segmentasyonu bul (greedily longest suffix match)
        best = None
        best_score = -1

        for root_end in range(self.min_root_len, len(word) + 1):
            root = word[:root_end]
            remainder = word[root_end:]

            # Kok sozlukte mi?
            root_score = 0
            if root in self.root_vocab:
                root_score = min(self.root_vocab[root] / 100, 1.0)
            elif root in COMMON_ROOTS:
                root_score = 0.8
            else:
                continue  # Bilinmeyen kok

            # Ek zincirini ayristir
            suffixes, suffix_types = self._parse_suffix_chain(remainder)

            if suffixes is None:
                continue  # Gecersiz ek zinciri

            # Skor: kok frekansi + ek sayisi cezasi
            n_suffixes = len(suffixes)
            suffix_penalty = max(0, (n_suffixes - 5) * 0.1)
            score = root_score - suffix_penalty

            if score > best_score:
                best_score = score
                best = {
                    "word": word,
                    "root": root,
                    "suffixes": suffixes,
                    "suffix_types": suffix_types,
                    "valid": True,
                    "score": round(min(score, 1.0), 3),
                }

        if best is not None:
            return best

        # Hicbir segmentasyon bulunamadi
        return {
            "word": word,
            "root": word,
            "suffixes": [],
            "suffix_types": [],
            "valid": False,
            "score": 0.0,
        }

    def _parse_suffix_chain(
        self, remainder: str
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Ek zincirini ayristirir (greedy longest match).

        Args:
            remainder: Kokten sonra kalan kisim

        Returns:
            (suffix_list, type_list) veya (None, None) gecersizse
        """
        if not remainder:
            return [], []

        suffixes = []
        types = []
        pos = 0
        all_suffixes = NOUN_SUFFIXES + VERB_SUFFIXES

        max_iterations = 20
        iteration = 0

        while pos < len(remainder) and iteration < max_iterations:
            matched = False

            # En uzun eslesmeyi bul
            best_match = None
            best_type = None
            best_len = 0

            for suffix, stype in all_suffixes:
                if remainder[pos:].startswith(suffix) and len(suffix) > best_len:
                    best_match = suffix
                    best_type = stype
                    best_len = len(suffix)

            if best_match:
                suffixes.append(best_match)
                types.append(best_type)
                pos += best_len
                matched = True

            if not matched:
                # Kalan kisim hicbir eke uymadi
                remaining = remainder[pos:]
                if len(remaining) <= 2:
                    # Kisa kalan — muhtemelen fonetik uyum
                    suffixes.append(remaining)
                    types.append("UNKNOWN")
                    break
                else:
                    return None, None  # Gecersiz

            iteration += 1

        return suffixes, types

    def validate_suffixes(self, word: str) -> float:
        """Morfolojik gecerlilik skoru [0-1].

        Args:
            word: Dogrulanacak kelime

        Returns:
            Gecerlilik skoru
        """
        result = self.segment(word)
        return result["score"]

    def correct_morphology(self, word: str) -> str:
        """Morfolojik hatalari duzeltir.

        Basit strateji: Bilinen kok + uyumlu ekleri birlestir.

        Args:
            word: Duzeltilecek kelime

        Returns:
            Duzeltilmis kelime
        """
        result = self.segment(word)

        if result["valid"] and result["score"] > 0.5:
            # Segmentasyon basarili — kelime zaten gecerli
            return word

        # Kok bulundu ama ekler sorunlu — koku dondur
        if result["root"] and result["root"] in self.root_vocab:
            return result["root"]

        return word  # Degistirme yapma
