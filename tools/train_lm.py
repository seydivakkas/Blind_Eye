"""
tools/train_lm.py
Türkçe N-Gram Dil Modeli Eğitimi — Saf Python ARPA Üretici

KenLM (lmplz) C++ bağımlılığı olmadan, saf Python ile
n-gram ARPA dil modeli eğitir.

Kullanım:
    python tools/train_lm.py --corpus data/corpus_tr.txt --order 3
    python tools/train_lm.py --dummy --order 3

Çıktı:
    models/tr_3gram.arpa  — ARPA formatında n-gram dil modeli

Corpus hazırlama:
    1. Türkçe Wikipedia dump'ı indir
    2. Zemberek ile normalize et (küçük harf, noktalama temizle)
    3. Her satır bir cümle olacak şekilde dosyaya yaz
"""

import os
import sys
import math
import argparse
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  1. METİN NORMALİZASYONU
# ═══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Türkçe metin normalizasyonu.

    - Küçük harfe çevir (Türkçe İ/I kuralları)
    - Noktalama temizle
    - Fazla boşlukları sil
    """
    tr_map = str.maketrans("İIÜÖÇŞĞ", "iıüöçşğ")
    text = text.translate(tr_map).lower()

    allowed = set("abcçdefgğhıijklmnoöprsştuüvyz ")
    text = "".join(c if c in allowed else " " for c in text)

    return " ".join(text.split())


def prepare_corpus(input_path: str, output_path: str) -> int:
    """Ham metin dosyasını normalize eder."""
    logger.info(f"Corpus normalize ediliyor: {input_path}")
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            normalized = normalize_text(line.strip())
            if len(normalized) > 5:
                fout.write(normalized + "\n")
                count += 1

    logger.info(f"  {count} satır normalize edildi → {output_path}")
    return count


# ═══════════════════════════════════════════════════════════════
#  2. SAF PYTHON N-GRAM ARPA ÜRETİCİ
# ═══════════════════════════════════════════════════════════════

class PythonNGramTrainer:
    """KenLM bağımlılığı olmadan saf Python ARPA modeli üretir.

    Witten-Bell smoothing ile n-gram olasılıkları hesaplar.
    Çıktı standart ARPA formatındadır ve pyctcdecode ile uyumludur.
    """

    BOS = "<s>"   # Beginning of sentence
    EOS = "</s>"   # End of sentence

    def __init__(self, order: int = 3):
        self.order = order
        self.ngram_counts: Dict[int, Counter] = {}
        self.total_tokens = 0
        self.vocab = set()

    def train(self, corpus_path: str):
        """Corpus dosyasından n-gram'ları çıkarır."""
        logger.info(f"N-gram eğitimi başlıyor (order={self.order})...")

        # N-gram sayaçlarını başlat
        for n in range(1, self.order + 1):
            self.ngram_counts[n] = Counter()

        line_count = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue

                # BOS/EOS ekle
                padded = [self.BOS] + tokens + [self.EOS]
                self.vocab.update(tokens)
                line_count += 1

                # Her n için n-gram'ları say
                for n in range(1, self.order + 1):
                    for i in range(len(padded) - n + 1):
                        ngram = tuple(padded[i:i + n])
                        self.ngram_counts[n][ngram] += 1

        self.total_tokens = sum(self.ngram_counts[1].values())
        self.vocab.add(self.BOS)
        self.vocab.add(self.EOS)

        logger.info(f"  {line_count} cümle işlendi")
        logger.info(f"  Vocab boyutu: {len(self.vocab)}")
        logger.info(f"  Toplam token: {self.total_tokens}")
        for n in range(1, self.order + 1):
            logger.info(f"  {n}-gram sayısı: {len(self.ngram_counts[n])}")

    def _log_prob(self, ngram: tuple) -> float:
        """Witten-Bell smoothing ile log10 olasılık hesaplar."""
        n = len(ngram)

        if n == 1:
            # Unigram: basit frekans
            count = self.ngram_counts[1].get(ngram, 0)
            if count == 0:
                return -5.0  # Bilinmeyen token için düşük olasılık
            return math.log10(count / self.total_tokens)

        # N > 1: Witten-Bell interpolasyon
        prefix = ngram[:-1]
        prefix_count = self.ngram_counts[n - 1].get(prefix, 0)

        if prefix_count == 0:
            # Prefix hiç görülmemiş → backoff
            return self._log_prob(ngram[1:])

        ngram_count = self.ngram_counts[n].get(ngram, 0)

        # Witten-Bell: T(prefix) = prefix'ten sonra gelen farklı kelime sayısı
        t_prefix = sum(
            1 for ng, c in self.ngram_counts[n].items()
            if ng[:-1] == prefix and c > 0
        )

        if ngram_count > 0:
            # P_ml = C(ngram) / C(prefix)
            # P_wb = (1 - λ) · P_ml + λ · P_backoff
            lam = t_prefix / (t_prefix + prefix_count)
            p_ml = ngram_count / prefix_count
            p_backoff = 10 ** self._log_prob(ngram[1:])
            prob = (1 - lam) * p_ml + lam * p_backoff
        else:
            # Görülmemiş n-gram → backoff
            lam = t_prefix / (t_prefix + prefix_count)
            prob = lam * (10 ** self._log_prob(ngram[1:]))

        return math.log10(max(prob, 1e-10))

    def _backoff_weight(self, prefix: tuple) -> float:
        """ARPA backoff ağırlığı (basitleştirilmiş)."""
        return -0.5  # Sabit backoff (Witten-Bell'de ayrı hesap gerektirmez)

    def write_arpa(self, output_path: str):
        """Standart ARPA formatında dil modeli yazar."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        logger.info(f"ARPA modeli yazılıyor → {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            # Header
            f.write("\\data\\\n")
            for n in range(1, self.order + 1):
                count = len(self.ngram_counts[n])
                f.write(f"ngram {n}={count}\n")
            f.write("\n")

            # N-gram bölümleri
            for n in range(1, self.order + 1):
                f.write(f"\\{n}-grams:\n")

                # Frekansa göre sırala (en sık ilk)
                sorted_ngrams = sorted(
                    self.ngram_counts[n].items(),
                    key=lambda x: -x[1]
                )

                for ngram, count in sorted_ngrams:
                    log_p = self._log_prob(ngram)
                    ngram_str = " ".join(ngram)

                    if n < self.order:
                        bow = self._backoff_weight(ngram)
                        f.write(f"{log_p:.4f}\t{ngram_str}\t{bow:.4f}\n")
                    else:
                        f.write(f"{log_p:.4f}\t{ngram_str}\n")

                f.write("\n")

            # Footer
            f.write("\\end\\\n")

        size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"✅ ARPA modeli yazıldı: {output_path} ({size_kb:.1f} KB)")


# ═══════════════════════════════════════════════════════════════
#  3. DUMMY CORPUS
# ═══════════════════════════════════════════════════════════════

def create_dummy_corpus(output_path: str, num_sentences: int = 1000):
    """Test amaçlı küçük Türkçe corpus oluşturur."""
    import random

    words = [
        "merhaba", "teşekkürler", "evet", "hayır", "başla", "durdur",
        "lütfen", "günaydın", "iyi", "akşamlar", "nasılsın", "ben",
        "sen", "biz", "siz", "ne", "nerede", "zaman", "gün", "bugün",
        "yarın", "tamam", "peki", "hoşça", "kal", "gel", "git",
        "yardım", "su", "ekmek", "çay", "kahve", "var", "yok",
        "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_sentences):
            length = random.randint(3, 8)
            sentence = " ".join(random.choices(words, k=length))
            f.write(sentence + "\n")

    logger.info(f"Dummy corpus oluşturuldu: {output_path} ({num_sentences} cümle)")


# ═══════════════════════════════════════════════════════════════
#  4. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Türkçe N-Gram Dil Modeli Eğitimi (Saf Python ARPA Üretici)"
    )
    parser.add_argument("--corpus", default="data/corpus_tr.txt",
                        help="Girdi metin dosyası (bir satır = bir cümle)")
    parser.add_argument("--output", default="models/tr_3gram.arpa",
                        help="ARPA çıktı dosyası")
    parser.add_argument("--order", type=int, default=3,
                        help="N-gram sırası (3 = trigram)")
    parser.add_argument("--dummy", action="store_true",
                        help="Test corpus oluştur (gerçek corpus yoksa)")

    args = parser.parse_args()

    if args.dummy:
        create_dummy_corpus(args.corpus)

    if not os.path.exists(args.corpus):
        logger.error(f"Corpus bulunamadı: {args.corpus}")
        logger.info("--dummy flag'i ile test corpus oluşturabilirsiniz.")
        sys.exit(1)

    # 1. Normalize
    norm_path = args.corpus + ".norm"
    prepare_corpus(args.corpus, norm_path)

    # 2. N-gram eğitimi (saf Python)
    trainer = PythonNGramTrainer(order=args.order)
    trainer.train(norm_path)

    # 3. ARPA yaz
    trainer.write_arpa(args.output)

    logger.info(f"\n🎉 Kullanım: LMDecoder(lm_path='{args.output}')")
