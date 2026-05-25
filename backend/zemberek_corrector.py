"""
backend/zemberek_corrector.py
─────────────────────────────
Türkçe NLP Son İşleme — Saf Python Yazım Düzeltici.

Amaç:
    CTC/Beam Search decoder'ın ürettiği ham metin çıktılarını Türkçe dil
    kurallarına göre düzeltir. Ağır Zemberek kütüphanesine bağımlı olmadan,
    Türkçe ünlü uyumu ve kelime frekansı tabanlı hafif bir post-processing
    katmanı sağlar.

Düzeltme Stratejisi:
    1. Sözlük Eşleme: data/corpus_tr.txt frekans sözlüğünde arama
    2. Ünlü Uyumu Kontrolü: Büyük/küçük ünlü uyumu doğrulama
    3. Levenshtein Düzeltme: En yakın sözlük kelimesine (frekans-ağırlıklı) eşleme

    Düzeltme Formülü:
        score(w_candidate) = (1 / (1 + distance)) × log(freq + 1) × vowel_harmony_bonus

    Burada:
        distance = Levenshtein(w_input, w_candidate)
        freq = Sözlükteki kelime frekansı
        vowel_harmony_bonus = 1.2 (uyumlu) veya 0.8 (uyumsuz)

Referanslar:
    - Oflazer (1994), "Two-level Description of Turkish Morphology"
    - Zemberek-NLP: https://github.com/ahmetaa/zemberek-nlp
    - TDK İmla Kılavuzu — Türkçe ünlü uyumu kuralları

Kullanım:
    from backend.zemberek_corrector import TurkishSpellChecker

    checker = TurkishSpellChecker()
    corrected = checker.correct("mrhaba")    # → "merhaba"
    corrected = checker.correct("günyaıdn")  # → "günaydın"

    # Tam cümle düzeltme
    result = checker.correct_sentence("mrhaba naslsn")
    # → "merhaba nasılsın"
"""

import os
import re
import math
import json
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Türkçe Fonetik Sabitleri
# ═══════════════════════════════════════════════

# Türkçe ünlüler
VOWELS_BACK = set("aıou")        # Kalın ünlüler (art)
VOWELS_FRONT = set("eiöü")       # İnce ünlüler (ön)
VOWELS_UNROUNDED = set("aeıi")   # Düz ünlüler
VOWELS_ROUNDED = set("oöuü")     # Yuvarlak ünlüler
ALL_VOWELS = VOWELS_BACK | VOWELS_FRONT

# Büyük ünlü uyumu tablosu (son ünlüye göre gelebilecek ünlüler)
GREAT_HARMONY = {
    "a": {"a", "ı"},
    "ı": {"a", "ı"},
    "o": {"a", "u"},
    "u": {"a", "u"},
    "e": {"e", "i"},
    "i": {"e", "i"},
    "ö": {"e", "ü"},
    "ü": {"e", "ü"},
}

# Küçük ünlü uyumu tablosu
SMALL_HARMONY = {
    "a": {"a", "e", "ı", "i"},
    "ı": {"a", "e", "ı", "i"},
    "o": {"a", "e", "u", "ü"},
    "u": {"a", "e", "u", "ü"},
    "e": {"a", "e", "ı", "i"},
    "i": {"a", "e", "ı", "i"},
    "ö": {"a", "e", "u", "ü"},
    "ü": {"a", "e", "u", "ü"},
}

# Türkçe'de yaygın hece kalıpları (CV, CVC, VC, V, CVCC)
TURKISH_SYLLABLE_PATTERN = re.compile(
    r"[bcçdfgğhjklmnprsştvyz]*[aeıioöuü][bcçdfgğhjklmnprsştvyz]*",
    re.IGNORECASE | re.UNICODE,
)


def _levenshtein(a: str, b: str) -> int:
    """İki string arasındaki Levenshtein mesafesini hesaplar.

    Tek satır DP — O(min(m,n)) bellek.
    """
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


class TurkishSpellChecker:
    """Türkçe Yazım Düzeltici — Saf Python.

    Sözlük + Ünlü Uyumu + Levenshtein tabanlı hafif post-processing.

    Args:
        corpus_path: Kelime frekans dosyası (her satırda kelimeler)
        vocab_path: Karakter seti JSON dosyası
        max_edit_distance: Maksimum düzenleme mesafesi eşiği
        min_word_freq: Minimum kelime frekansı eşiği
    """

    def __init__(
        self,
        corpus_path: str = "data/corpus_tr.txt",
        vocab_path: str = "configs/vocab.json",
        max_edit_distance: int = 3,
        min_word_freq: int = 1,
    ):
        self.max_edit_distance = max_edit_distance
        self.min_word_freq = min_word_freq

        # Kelime frekans sözlüğü
        self.word_freq: Dict[str, int] = {}
        self._load_corpus(corpus_path)

        # Karakter seti
        self.valid_chars = self._load_charset(vocab_path)

        # İstatistikler
        self._corrections_made = 0
        self._total_words = 0

        logger.info(
            f"TurkishSpellChecker hazır: "
            f"{len(self.word_freq)} kelime sözlüğü, "
            f"max_edit={max_edit_distance}"
        )

    def _load_corpus(self, path: str):
        """Kelime frekans sözlüğünü corpus dosyasından oluşturur."""
        if not os.path.exists(path):
            logger.warning(f"{path} bulunamadı — boş sözlük kullanılacak.")
            return

        word_counter = Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().lower().split()
                word_counter.update(words)

        # Minimum frekans filtresi
        self.word_freq = {
            word: freq
            for word, freq in word_counter.items()
            if freq >= self.min_word_freq
        }

        logger.info(
            f"Corpus yüklendi: {sum(word_counter.values())} token, "
            f"{len(self.word_freq)} benzersiz kelime"
        )

    def _load_charset(self, path: str) -> set:
        """Geçerli karakter setini yükler."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chars = set()
            for ch in data.get("charset", []):
                if ch != "<blank>":
                    chars.add(ch)
            return chars
        except FileNotFoundError:
            return set("abcçdefgğhıijklmnoöprsştuüvyz ")

    # ═══════════════════════════════════════
    #  Ünlü Uyumu Kontrolü
    # ═══════════════════════════════════════

    def check_great_vowel_harmony(self, word: str) -> Tuple[bool, float]:
        """Büyük ünlü uyumu (kalın-ince uyumu) kontrolü.

        Kural: Bir kelimedeki tüm ünlüler ya kalın ya da ince olmalıdır.
        Yani son ünlü kalınsa sonraki de kalın, inceyse ince olmalıdır.

        Returns:
            (uyumlu_mu, uyum_oranı) tuple'ı
        """
        vowels_in_word = [ch for ch in word.lower() if ch in ALL_VOWELS]

        if len(vowels_in_word) <= 1:
            return True, 1.0

        violations = 0
        for i in range(1, len(vowels_in_word)):
            prev = vowels_in_word[i - 1]
            curr = vowels_in_word[i]
            if prev in GREAT_HARMONY:
                if curr not in GREAT_HARMONY[prev]:
                    violations += 1

        total_transitions = len(vowels_in_word) - 1
        harmony_ratio = 1.0 - (violations / total_transitions)
        return violations == 0, harmony_ratio

    def check_small_vowel_harmony(self, word: str) -> Tuple[bool, float]:
        """Küçük ünlü uyumu (düz-yuvarlak uyumu) kontrolü.

        Returns:
            (uyumlu_mu, uyum_oranı) tuple'ı
        """
        vowels_in_word = [ch for ch in word.lower() if ch in ALL_VOWELS]

        if len(vowels_in_word) <= 1:
            return True, 1.0

        violations = 0
        for i in range(1, len(vowels_in_word)):
            prev = vowels_in_word[i - 1]
            curr = vowels_in_word[i]
            if prev in SMALL_HARMONY:
                if curr not in SMALL_HARMONY[prev]:
                    violations += 1

        total_transitions = len(vowels_in_word) - 1
        harmony_ratio = 1.0 - (violations / total_transitions)
        return violations == 0, harmony_ratio

    def vowel_harmony_score(self, word: str) -> float:
        """Birleşik ünlü uyumu skoru [0.0, 1.0].

        score = 0.6 × great_harmony + 0.4 × small_harmony
        """
        _, great = self.check_great_vowel_harmony(word)
        _, small = self.check_small_vowel_harmony(word)
        return 0.6 * great + 0.4 * small

    # ═══════════════════════════════════════
    #  Kelime Düzeltme
    # ═══════════════════════════════════════

    def correct(self, word: str) -> str:
        """Tek kelimeyi düzeltir.

        Düzeltme Adımları:
            1. Sözlükte varsa → doğrudan döndür
            2. Sözlükte yoksa → en yakın adayı bul
            3. Aday bulunamazsa → orijinal kelimeyi döndür

        Args:
            word: Düzeltilecek kelime

        Returns:
            Düzeltilmiş kelime
        """
        self._total_words += 1
        word_lower = word.strip().lower()

        if not word_lower:
            return word

        # 1. Sözlükte doğrudan var mı?
        if word_lower in self.word_freq:
            return word_lower

        # 2. En yakın adayı bul
        best_candidate = self._find_best_candidate(word_lower)
        if best_candidate is not None:
            self._corrections_made += 1
            return best_candidate

        # 3. Bulunamazsa orijinal döndür
        return word_lower

    def _find_best_candidate(self, word: str) -> Optional[str]:
        """Frekans-ağırlıklı Levenshtein ile en iyi düzeltme adayını bulur.

        score = (1 / (1 + distance)) × log(freq + 1) × vowel_bonus

        Returns:
            En iyi düzeltme adayı veya None
        """
        best_score = -1.0
        best_candidate = None

        # Uzunluk filtresi: çok uzun/kısa kelimeleri atla
        min_len = max(1, len(word) - self.max_edit_distance)
        max_len = len(word) + self.max_edit_distance

        for candidate, freq in self.word_freq.items():
            # Hızlı uzunluk filtresi
            if not (min_len <= len(candidate) <= max_len):
                continue

            # İlk karakter heuristic — genellikle ilk harf doğrudur
            # Bu çok katı olabilir, bu yüzden sadece çok uzak adayları atla
            distance = _levenshtein(word, candidate)

            if distance > self.max_edit_distance:
                continue

            if distance == 0:
                return candidate  # Tam eşleşme

            # Skor hesapla
            proximity = 1.0 / (1.0 + distance)
            freq_score = math.log(freq + 1)
            vowel_bonus = 1.2 if self.vowel_harmony_score(candidate) > 0.8 else 0.8

            score = proximity * freq_score * vowel_bonus

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def correct_sentence(self, sentence: str) -> str:
        """Tam cümleyi düzeltir — her kelime bağımsız düzeltilir.

        Args:
            sentence: Düzeltilecek cümle

        Returns:
            Düzeltilmiş cümle
        """
        words = sentence.strip().split()
        corrected = [self.correct(w) for w in words]
        return " ".join(corrected)

    def correct_with_details(self, word: str) -> Dict:
        """Düzeltme detayları ile birlikte döndürür (debug/analiz).

        Returns:
            {
                "input": orijinal kelime,
                "output": düzeltilmiş kelime,
                "in_vocabulary": sözlükte mi,
                "corrected": düzeltme yapıldı mı,
                "edit_distance": düzeltme mesafesi,
                "frequency": sözlük frekansı,
                "vowel_harmony": ünlü uyumu skoru,
                "candidates": en iyi 5 aday listesi
            }
        """
        word_lower = word.strip().lower()
        in_vocab = word_lower in self.word_freq

        result = {
            "input": word,
            "output": word_lower,
            "in_vocabulary": in_vocab,
            "corrected": False,
            "edit_distance": 0,
            "frequency": self.word_freq.get(word_lower, 0),
            "vowel_harmony": self.vowel_harmony_score(word_lower),
            "candidates": [],
        }

        if in_vocab:
            return result

        # Aday listesi oluştur
        candidates = []
        min_len = max(1, len(word_lower) - self.max_edit_distance)
        max_len = len(word_lower) + self.max_edit_distance

        for candidate, freq in self.word_freq.items():
            if not (min_len <= len(candidate) <= max_len):
                continue

            distance = _levenshtein(word_lower, candidate)
            if distance <= self.max_edit_distance:
                proximity = 1.0 / (1.0 + distance)
                freq_score = math.log(freq + 1)
                vowel_bonus = 1.2 if self.vowel_harmony_score(candidate) > 0.8 else 0.8
                score = proximity * freq_score * vowel_bonus

                candidates.append({
                    "word": candidate,
                    "distance": distance,
                    "frequency": freq,
                    "score": round(score, 4),
                    "vowel_harmony": round(self.vowel_harmony_score(candidate), 3),
                })

        # Skora göre sırala
        candidates.sort(key=lambda x: -x["score"])
        result["candidates"] = candidates[:5]

        if candidates:
            best = candidates[0]
            result["output"] = best["word"]
            result["corrected"] = True
            result["edit_distance"] = best["distance"]

        return result

    def get_statistics(self) -> Dict:
        """Düzeltme istatistiklerini döndürür."""
        return {
            "total_words": self._total_words,
            "corrections_made": self._corrections_made,
            "correction_rate": (
                self._corrections_made / self._total_words * 100
                if self._total_words > 0
                else 0.0
            ),
            "vocabulary_size": len(self.word_freq),
            "top_10_frequent": dict(
                Counter(self.word_freq).most_common(10)
            ),
        }

    def reset_statistics(self):
        """İstatistikleri sıfırlar."""
        self._corrections_made = 0
        self._total_words = 0


class PostProcessor:
    """CTC Decoder çıktısı için son işleme pipeline'ı.

    Bileşenler:
        1. Karakter temizleme (geçersiz karakterleri kaldır)
        2. CTC artefakt düzeltme (tekrarlayan karakterler)
        3. Türkçe yazım düzeltme (TurkishSpellChecker)
        4. Ünlü uyumu doğrulama

    Kullanım:
        post = PostProcessor()
        clean = post.process("mrrrhba")  # → "merhaba"
    """

    def __init__(
        self,
        corpus_path: str = "data/corpus_tr.txt",
        vocab_path: str = "configs/vocab.json",
        max_edit_distance: int = 2,
    ):
        self.spell_checker = TurkishSpellChecker(
            corpus_path=corpus_path,
            vocab_path=vocab_path,
            max_edit_distance=max_edit_distance,
        )

        # CTC artefakt kalıpları
        self._patterns = [
            (re.compile(r"(.)\1{2,}"), r"\1"),      # 3+ tekrar → 1
            (re.compile(r"\s{2,}"), " "),             # Çoklu boşluk → tek
            (re.compile(r"[^\w\s]", re.UNICODE), ""), # Özel karakterleri kaldır
        ]

    def process(self, text: str) -> str:
        """Tam son işleme pipeline'ı uygular.

        Args:
            text: CTC decoder ham çıktısı

        Returns:
            Temizlenmiş ve düzeltilmiş metin
        """
        if not text or not text.strip():
            return ""

        # 1. CTC artefakt temizleme
        cleaned = text.lower().strip()
        for pattern, repl in self._patterns:
            cleaned = pattern.sub(repl, cleaned)

        # 2. Kelime bazlı Türkçe düzeltme
        corrected = self.spell_checker.correct_sentence(cleaned)

        return corrected.strip()

    def process_with_confidence(
        self, text: str, ctc_confidence: float
    ) -> Tuple[str, float]:
        """Güven skoru ile birlikte son işleme.

        Düzeltme yapıldıysa güven düşer (model yanlış tahmin etmiş demektir).

        Returns:
            (düzeltilmiş_metin, ayarlanmış_güven) tuple'ı
        """
        original = text
        processed = self.process(text)

        # Düzeltme mesafesine göre güven düşürme
        if original.strip().lower() != processed:
            distance = _levenshtein(original.strip().lower(), processed)
            confidence_penalty = min(0.15 * distance, 0.5)
            adjusted_confidence = max(0.0, ctc_confidence - confidence_penalty)
        else:
            adjusted_confidence = ctc_confidence

        return processed, adjusted_confidence
