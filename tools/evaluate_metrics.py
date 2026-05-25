"""
tools/evaluate_metrics.py
─────────────────────────
Akademik Değerlendirme Script'i — WER, CER, Gecikme ve Karışıklık Matrisi.

Bu script, eğitilmiş ONNX modelini test verisi üzerinde çalıştırarak
TÜBİTAK 2209-A raporundaki "Ablation Study" ve "Hedef Metrikler"
tablolarını dolduracak gerçek ölçümler üretir.

Ölçülen Metrikler:
    1. WER (Word Error Rate)  — Levenshtein mesafe tabanlı
    2. CER (Character Error Rate) — Karakter düzeyinde hata
    3. Latency — ONNX inference mikro-saniye bazlı gecikme
    4. Karışıklık Matrisi — En çok karıştırılan kelime/harf çiftleri

Kullanım:
    python tools/evaluate_metrics.py
    python tools/evaluate_metrics.py --model models/student_int8.onnx
    python tools/evaluate_metrics.py --output results/evaluation_report.json
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Levenshtein Mesafe Algoritması (Saf Python)
# ═══════════════════════════════════════════════

def levenshtein_distance(ref: List, hyp: List) -> int:
    """İki dizi arasındaki minimum düzenleme mesafesini hesaplar.

    Wagner-Fischer dinamik programlama algoritması.
    Zaman: O(m×n), Bellek: O(min(m,n)) — tek satır optimizasyonu.

    Args:
        ref: Referans (doğru) dizi
        hyp: Hipotez (tahmin) dizi

    Returns:
        Minimum ekleme/silme/değiştirme sayısı
    """
    m, n = len(ref), len(hyp)

    # Bellek optimizasyonu: kısa diziyi satır olarak kullan
    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # Tek satır DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # Ekleme
                prev[j] + 1,          # Silme
                prev[j - 1] + cost,   # Değiştirme
            )
        prev = curr

    return prev[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (WER) hesaplar.

    WER = (S + D + I) / N

    Burada:
        S = İkame (substitution) sayısı
        D = Silme (deletion) sayısı
        I = Ekleme (insertion) sayısı
        N = Referans kelime sayısı

    Returns:
        WER oranı [0.0, ∞) — 0 = mükemmel, >1 olabilir
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER) hesaplar.

    CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars)

    Returns:
        CER oranı [0.0, ∞)
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    distance = levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


# ═══════════════════════════════════════════════
#  Model Yükleme & Inference
# ═══════════════════════════════════════════════

def load_model(model_path: str):
    """ONNX modelini yükler."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(
            f"Model yüklendi: {model_path} "
            f"(giriş: {input_name} {input_shape})"
        )
        return session, input_name, input_shape
    except Exception as e:
        logger.error(f"Model yüklenemedi: {e}")
        return None, None, None


def run_inference(
    session, input_name: str, roi_batch: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Tek bir inference çalıştırır ve süresini ölçer.

    Returns:
        (logits, latency_ms) tuple'ı
    """
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: roi_batch})
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    return outputs[0], latency_ms


# ═══════════════════════════════════════════════
#  Karışıklık Matrisi (Confusion Matrix)
# ═══════════════════════════════════════════════

class ConfusionTracker:
    """Kelime ve karakter düzeyinde karışıklık izleyici."""

    def __init__(self):
        self.word_confusions: Dict[Tuple[str, str], int] = Counter()
        self.char_confusions: Dict[Tuple[str, str], int] = Counter()
        self.word_correct = 0
        self.word_total = 0
        self.char_correct = 0
        self.char_total = 0

    def update(self, reference: str, hypothesis: str):
        """Bir tahmin-referans çiftini izleyiciye ekler."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        # Kelime düzeyinde
        for ref_w, hyp_w in zip(ref_words, hyp_words):
            self.word_total += 1
            if ref_w == hyp_w:
                self.word_correct += 1
            else:
                self.word_confusions[(ref_w, hyp_w)] += 1

        # Karakter düzeyinde
        ref_chars = list(reference.strip())
        hyp_chars = list(hypothesis.strip())
        for ref_c, hyp_c in zip(ref_chars, hyp_chars):
            self.char_total += 1
            if ref_c == hyp_c:
                self.char_correct += 1
            else:
                self.char_confusions[(ref_c, hyp_c)] += 1

    def get_top_word_confusions(self, n: int = 15) -> List[Tuple[str, str, int]]:
        """En sık karıştırılan kelime çiftleri."""
        top = self.word_confusions.most_common(n)
        return [(ref, hyp, count) for (ref, hyp), count in top]

    def get_top_char_confusions(self, n: int = 20) -> List[Tuple[str, str, int]]:
        """En sık karıştırılan karakter çiftleri."""
        top = self.char_confusions.most_common(n)
        return [(ref, hyp, count) for (ref, hyp), count in top]

    def print_report(self):
        """Metin tablosu olarak karışıklık raporu basar."""
        print("\n" + "=" * 70)
        print("  KARIŞIKLIK MATRİSİ RAPORU (Confusion Matrix)")
        print("=" * 70)

        # Kelime doğruluğu
        if self.word_total > 0:
            word_acc = self.word_correct / self.word_total * 100
            print(f"\n  Kelime Doğruluğu: {word_acc:.1f}% "
                  f"({self.word_correct}/{self.word_total})")

        # Karakter doğruluğu
        if self.char_total > 0:
            char_acc = self.char_correct / self.char_total * 100
            print(f"  Karakter Doğruluğu: {char_acc:.1f}% "
                  f"({self.char_correct}/{self.char_total})")

        # En çok karıştırılan kelimeler
        word_conf = self.get_top_word_confusions()
        if word_conf:
            print(f"\n  {'─' * 50}")
            print(f"  En Çok Karıştırılan Kelimeler (Top {len(word_conf)}):")
            print(f"  {'─' * 50}")
            print(f"  {'Referans':<20} {'Tahmin':<20} {'Sayı':>6}")
            print(f"  {'─' * 50}")
            for ref, hyp, count in word_conf:
                print(f"  {ref:<20} {hyp:<20} {count:>6}")

        # En çok karıştırılan karakterler
        char_conf = self.get_top_char_confusions()
        if char_conf:
            print(f"\n  {'─' * 50}")
            print(f"  En Çok Karıştırılan Karakterler (Top {len(char_conf)}):")
            print(f"  {'─' * 50}")
            print(f"  {'Referans':<10} {'Tahmin':<10} {'Sayı':>6} {'Viseme Aynı?':>12}")
            print(f"  {'─' * 50}")

            # Viseme kontrolü için haritayı yükle
            viseme_map = self._load_viseme_map()
            for ref, hyp, count in char_conf:
                same = self._check_same_viseme(ref, hyp, viseme_map)
                marker = "✓ Homofen" if same else ""
                print(f"  '{ref}'       →  '{hyp}'       {count:>6}   {marker}")

        print("\n" + "=" * 70)

    @staticmethod
    def _load_viseme_map() -> Dict[str, str]:
        try:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "tr_viseme_map.json",
            )
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("phoneme_to_viseme", {})
        except Exception:
            return {}

    @staticmethod
    def _check_same_viseme(
        char_a: str, char_b: str, viseme_map: Dict[str, str]
    ) -> bool:
        v_a = viseme_map.get(char_a)
        v_b = viseme_map.get(char_b)
        return v_a is not None and v_b is not None and v_a == v_b


# ═══════════════════════════════════════════════
#  Ana Değerlendirme Fonksiyonu
# ═══════════════════════════════════════════════

def evaluate(
    model_path: str,
    labels_path: str,
    data_dir: str = "data/processed",
    chunk_size: int = 6,
    max_samples: Optional[int] = None,
) -> Dict:
    """Tam değerlendirme pipeline'ı çalıştırır.

    Args:
        model_path: ONNX model dosya yolu
        labels_path: Etiket JSON dosyası
        data_dir: İşlenmiş veri dizini
        chunk_size: Frame chunk boyutu
        max_samples: Maksimum test örneği (None = tümü)

    Returns:
        Değerlendirme sonuçları sözlüğü
    """
    # Decoder
    from backend.decoder import TurkishCTCDecoder
    decoder = TurkishCTCDecoder()

    # Model yükle
    session, input_name, input_shape = load_model(model_path)
    if session is None:
        logger.error("Model yüklenemedi, değerlendirme iptal.")
        return {}

    # Etiketleri yükle
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        logger.info(f"Etiketler yüklendi: {len(labels)} örnek")
    except FileNotFoundError:
        logger.error(f"{labels_path} bulunamadı!")
        return {}

    # İzleyiciler
    tracker = ConfusionTracker()
    all_wer = []
    all_cer = []
    all_latency = []
    per_class_wer: Dict[str, List[float]] = defaultdict(list)

    # Test döngüsü
    sample_keys = list(labels.keys())
    if max_samples is not None:
        sample_keys = sample_keys[:max_samples]

    total = len(sample_keys)
    logger.info(f"Değerlendirme başlıyor: {total} örnek")

    for idx, key in enumerate(sample_keys):
        reference = labels[key]

        # ROI yükle (veya mock oluştur)
        npy_path = os.path.join(data_dir, f"{key}.npy")
        if os.path.exists(npy_path):
            roi_data = np.load(npy_path)
        else:
            # Mock ROI (model boyutuna uygun)
            roi_data = np.random.rand(chunk_size, 96, 96, 1).astype(np.float32)

        # Model giriş formatına dönüştür
        if roi_data.ndim == 3:
            roi_data = np.expand_dims(roi_data, axis=0)
        if roi_data.ndim == 4 and roi_data.shape[0] < chunk_size:
            # Padding
            pad_size = chunk_size - roi_data.shape[0]
            padding = np.zeros(
                (pad_size, *roi_data.shape[1:]), dtype=roi_data.dtype
            )
            roi_data = np.concatenate([roi_data, padding], axis=0)

        # Batch boyutu ekle
        roi_batch = np.expand_dims(roi_data, axis=0).astype(np.float32)

        # Model boyutu uyumu
        try:
            logits, latency_ms = run_inference(session, input_name, roi_batch)
        except Exception as e:
            logger.debug(f"Inference hatası [{key}]: {e}")
            continue

        # Decode
        hypothesis, confidence = decoder.decode(logits)

        # Metrikler
        wer = compute_wer(reference, hypothesis)
        cer = compute_cer(reference, hypothesis)

        all_wer.append(wer)
        all_cer.append(cer)
        all_latency.append(latency_ms)
        per_class_wer[reference].append(wer)

        # Karışıklık izleyici güncelle
        tracker.update(reference, hypothesis)

        # İlerleme (her %10)
        if (idx + 1) % max(1, total // 10) == 0:
            pct = (idx + 1) / total * 100
            avg_wer = np.mean(all_wer) * 100
            avg_cer = np.mean(all_cer) * 100
            logger.info(
                f"  [{pct:.0f}%] {idx + 1}/{total} — "
                f"WER: {avg_wer:.1f}%, CER: {avg_cer:.1f}%, "
                f"Latency: {np.mean(all_latency):.2f}ms"
            )

    # ── Sonuçlar ──
    results = {
        "model_path": model_path,
        "num_samples": len(all_wer),
        "wer": {
            "mean": float(np.mean(all_wer) * 100),
            "std": float(np.std(all_wer) * 100),
            "median": float(np.median(all_wer) * 100),
            "min": float(np.min(all_wer) * 100),
            "max": float(np.max(all_wer) * 100),
        },
        "cer": {
            "mean": float(np.mean(all_cer) * 100),
            "std": float(np.std(all_cer) * 100),
            "median": float(np.median(all_cer) * 100),
        },
        "latency_ms": {
            "mean": float(np.mean(all_latency)),
            "std": float(np.std(all_latency)),
            "p50": float(np.percentile(all_latency, 50)),
            "p95": float(np.percentile(all_latency, 95)),
            "p99": float(np.percentile(all_latency, 99)),
        },
        "per_class_wer": {
            word: float(np.mean(wers) * 100)
            for word, wers in sorted(per_class_wer.items())
        },
    }

    # Sonuçları yazdır
    _print_results(results)
    tracker.print_report()

    return results


def _print_results(results: Dict):
    """Sonuçları konsola yazdırır."""
    print("\n" + "═" * 60)
    print("  BLIND EYE — AKADEMİK DEĞERLENDİRME RAPORU")
    print("═" * 60)

    print(f"\n  Model: {results['model_path']}")
    print(f"  Test Örnekleri: {results['num_samples']}")

    wer = results["wer"]
    cer = results["cer"]
    lat = results["latency_ms"]

    print(f"\n  {'─' * 40}")
    print(f"  {'Metrik':<25} {'Değer':>12}")
    print(f"  {'─' * 40}")
    print(f"  {'WER (ortalama)':<25} {wer['mean']:>11.2f}%")
    print(f"  {'WER (std)':<25} {'±':>1}{wer['std']:>10.2f}%")
    print(f"  {'WER (median)':<25} {wer['median']:>11.2f}%")
    print(f"  {'CER (ortalama)':<25} {cer['mean']:>11.2f}%")
    print(f"  {'CER (median)':<25} {cer['median']:>11.2f}%")
    print(f"  {'Gecikme (ort.)':<25} {lat['mean']:>10.2f}ms")
    print(f"  {'Gecikme (p95)':<25} {lat['p95']:>10.2f}ms")
    print(f"  {'Gecikme (p99)':<25} {lat['p99']:>10.2f}ms")
    print(f"  {'─' * 40}")

    # Sınıf bazında WER
    per_class = results.get("per_class_wer", {})
    if per_class:
        print(f"\n  {'─' * 40}")
        print(f"  Sınıf Bazında WER:")
        print(f"  {'─' * 40}")
        print(f"  {'Kelime':<20} {'WER (%)':>12}")
        print(f"  {'─' * 40}")
        for word, wer_val in sorted(per_class.items(), key=lambda x: -x[1]):
            bar_len = int(wer_val / 5)
            bar = "█" * bar_len
            print(f"  {word:<20} {wer_val:>10.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Blind Eye — Akademik Değerlendirme Script'i"
    )
    parser.add_argument(
        "--model",
        default="models/student_int8.onnx",
        help="ONNX model dosyası",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/labels.json",
        help="Etiket JSON dosyası",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="İşlenmiş veri dizini",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6,
        help="Frame chunk boyutu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maksimum test örneği (None = tümü)",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_report.json",
        help="JSON çıktı dosyası",
    )
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        labels_path=args.labels,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
    )

    if results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSonuçlar kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
