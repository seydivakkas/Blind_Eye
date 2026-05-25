"""
tools/split_dataset.py
──────────────────────
Konusmaci-Bagimsiz Veri Bolunmesi (Speaker-Independent Split).

Akademik Motivasyon:
    Standart random split, ayni konusmacinin kliplerini hem egitim hem
    test setine dagitabilir. Bu durumda model "konusmaci yuzunu" ogrenip
    "dudak okuma" yapmak yerine "kisi tanima" yapar. Speaker-independent
    split, test setindeki tum konusmacilarin egitimde hic gorulmemesini
    garanti eder.

Algoritmasi:
    1. Dosya adlarindan konusmaci ID cikar (regex)
    2. Konusmaci ID bazli stratified split: %70 train / %15 val / %15 test
    3. Ayni konusmacinin hicbir klibi farkli setlere sizmaz
    4. Generalization gap: WER_unseen - WER_seen

Kullanim:
    python tools/split_dataset.py
    python tools/split_dataset.py --data-dir data/processed --seed 42
    python tools/split_dataset.py --output results/speaker_splits.json
"""

import os
import sys
import re
import json
import random
import logging
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================================================
#  Konusmaci ID Cikarma Strategileri
# ======================================================

# Dosya adi desenleri (onceligi yukari)
SPEAKER_PATTERNS = [
    re.compile(r"subj(\d+)_"),           # subj01_merhaba_01.npy
    re.compile(r"speaker(\d+)_"),        # speaker01_merhaba.npy
    re.compile(r"spk(\d+)_"),            # spk01_hello.npy
    re.compile(r"s(\d+)_"),              # s01_merhaba.npy
    re.compile(r"^(\d+)_"),              # 01_merhaba.npy
    re.compile(r"_s(\d+)\."),            # merhaba_s01.npy
    re.compile(r"_subj(\d+)\."),         # merhaba_subj01.npy
]


def extract_speaker_id(filename: str) -> Optional[str]:
    """Dosya adindan konusmaci ID cikarir.

    Birden fazla desen denenir, ilk eslesen kullanilir.

    Args:
        filename: .npy dosya adi (uzantisiz veya uzantili)

    Returns:
        Konusmaci ID string'i veya None
    """
    basename = os.path.basename(filename)

    for pattern in SPEAKER_PATTERNS:
        match = pattern.search(basename)
        if match:
            return f"speaker_{match.group(1).zfill(3)}"

    return None


def extract_word_label(filename: str) -> str:
    """Dosya adindan kelime etiketini cikarir.

    Ornek: subj01_merhaba_03.npy -> merhaba
    """
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Sayisal oneki ve soneki kaldir
    parts = basename.split("_")
    word_parts = []
    for part in parts:
        if not re.match(r"^(subj|speaker|spk|s)?\d+$", part, re.IGNORECASE):
            word_parts.append(part)

    return "_".join(word_parts) if word_parts else basename


class SpeakerSplitter:
    """Konusmaci-bagimsiz veri bolunme motoru.

    Konusmaci ID bazli katmanlı (stratified) bolunme yapar.
    Ayni konusmacinin tum klipleri ayni sette kalir.

    Args:
        data_dir: Islemenmis veri dizini (.npy dosyalari)
        labels_path: Etiket JSON dosyasi
        train_ratio: Egitim seti orani (varsayilan: 0.70)
        val_ratio: Dogrulama seti orani (varsayilan: 0.15)
        seed: Rastgelelik tohumu (varsayilan: 42)
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        labels_path: str = "data/processed/labels.json",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.labels_path = labels_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

        # Konusmaci -> dosya listesi
        self.speaker_files: Dict[str, List[str]] = defaultdict(list)
        # Dosya -> konusmaci
        self.file_speaker: Dict[str, str] = {}
        # Etiketler
        self.labels: Dict[str, str] = {}

        self._scan_dataset()

    def _scan_dataset(self):
        """Veri setini tarayarak konusmaci-dosya haritasini olusturur."""
        # Etiketleri yukle
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            logger.info(f"Etiketler yuklendi: {len(self.labels)} ornek")

        # .npy dosyalarini tara
        npy_files = []
        if os.path.exists(self.data_dir):
            npy_files = [
                f for f in os.listdir(self.data_dir)
                if f.endswith(".npy")
            ]

        if not npy_files and self.labels:
            # Etiketlerden dosya listesi olustur
            npy_files = [f"{key}.npy" for key in self.labels.keys()]

        # Konusmaci ID cikar
        unidentified = 0
        for npy_file in npy_files:
            speaker_id = extract_speaker_id(npy_file)
            key = os.path.splitext(npy_file)[0]

            if speaker_id is None:
                unidentified += 1
                # Tanimsiz konusmacilari hash ile gruplama
                # (kaba yontem: kelime bazli gruplama)
                word = extract_word_label(npy_file)
                speaker_id = f"unknown_{hash(key) % 100:03d}"

            self.speaker_files[speaker_id].append(key)
            self.file_speaker[key] = speaker_id

        total_speakers = len(self.speaker_files)
        total_files = sum(len(v) for v in self.speaker_files.values())

        if unidentified > 0:
            logger.warning(
                f"{unidentified}/{total_files} dosyadan konusmaci ID cikarilamadi. "
                f"Hash-bazli gruplama kullanildi."
            )

        logger.info(
            f"Veri seti taramasi: {total_files} dosya, "
            f"{total_speakers} konusmaci tespit edildi"
        )

    def split(self) -> Dict[str, List[str]]:
        """Konusmaci-bagimsiz bolunme yapar.

        Algoritma:
            1. Konusmaci ID listesini karistir (fixed seed)
            2. %70 train, %15 val, %15 test olarak bol
            3. Her setteki dosyalari dondur

        Returns:
            {"train": [...], "val": [...], "test": [...],
             "train_speakers": [...], "val_speakers": [...], "test_speakers": [...]}
        """
        speaker_ids = sorted(self.speaker_files.keys())
        n_speakers = len(speaker_ids)

        if n_speakers < 3:
            logger.warning(
                f"Sadece {n_speakers} konusmaci var — "
                f"minimum 3 konusmaci gerekli. Kelime bazli split yapilacak."
            )
            return self._fallback_word_split()

        # Karistir (fixed seed)
        rng = random.Random(self.seed)
        rng.shuffle(speaker_ids)

        # Bolunme indeksleri
        n_train = max(1, int(n_speakers * self.train_ratio))
        n_val = max(1, int(n_speakers * self.val_ratio))
        n_test = n_speakers - n_train - n_val

        if n_test < 1:
            n_val = max(1, n_val - 1)
            n_test = n_speakers - n_train - n_val

        train_speakers = speaker_ids[:n_train]
        val_speakers = speaker_ids[n_train:n_train + n_val]
        test_speakers = speaker_ids[n_train + n_val:]

        # Dosyalari topla
        train_files = []
        val_files = []
        test_files = []

        for spk in train_speakers:
            train_files.extend(self.speaker_files[spk])
        for spk in val_speakers:
            val_files.extend(self.speaker_files[spk])
        for spk in test_speakers:
            test_files.extend(self.speaker_files[spk])

        result = {
            "train": sorted(train_files),
            "val": sorted(val_files),
            "test": sorted(test_files),
            "train_speakers": sorted(train_speakers),
            "val_speakers": sorted(val_speakers),
            "test_speakers": sorted(test_speakers),
            "metadata": {
                "seed": self.seed,
                "n_speakers": n_speakers,
                "n_train_speakers": len(train_speakers),
                "n_val_speakers": len(val_speakers),
                "n_test_speakers": len(test_speakers),
                "n_train_files": len(train_files),
                "n_val_files": len(val_files),
                "n_test_files": len(test_files),
                "split_ratios": {
                    "train": self.train_ratio,
                    "val": self.val_ratio,
                    "test": self.test_ratio,
                },
            },
        }

        self._verify_no_leakage(result)
        self._print_split_summary(result)

        return result

    def _fallback_word_split(self) -> Dict[str, List[str]]:
        """Konusmaci yetersizse kelime bazli split yapar."""
        all_files = sorted(
            f for files in self.speaker_files.values() for f in files
        )
        rng = random.Random(self.seed)
        rng.shuffle(all_files)

        n = len(all_files)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        return {
            "train": sorted(all_files[:n_train]),
            "val": sorted(all_files[n_train:n_train + n_val]),
            "test": sorted(all_files[n_train + n_val:]),
            "train_speakers": ["fallback"],
            "val_speakers": ["fallback"],
            "test_speakers": ["fallback"],
            "metadata": {
                "seed": self.seed,
                "fallback": True,
                "reason": "Yetersiz konusmaci sayisi",
            },
        }

    def _verify_no_leakage(self, splits: Dict):
        """Konusmaci sizintisi olmadigini dogrular."""
        train_spk = set(splits["train_speakers"])
        val_spk = set(splits["val_speakers"])
        test_spk = set(splits["test_speakers"])

        assert train_spk.isdisjoint(val_spk), "Train-Val konusmaci sizintisi!"
        assert train_spk.isdisjoint(test_spk), "Train-Test konusmaci sizintisi!"
        assert val_spk.isdisjoint(test_spk), "Val-Test konusmaci sizintisi!"

        # Dosya bazli da kontrol
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert train_set.isdisjoint(val_set), "Train-Val dosya sizintisi!"
        assert train_set.isdisjoint(test_set), "Train-Test dosya sizintisi!"

        logger.info("Sizinti kontrolu GECTI — konusmaci izolasyonu saglandi.")

    def _print_split_summary(self, splits: Dict):
        """Bolunme ozetini yazdirir."""
        meta = splits["metadata"]
        print("\n" + "=" * 60)
        print("  KONUSMACI-BAGIMSIZ BOLUNME OZETI")
        print("=" * 60)
        print(f"\n  Toplam konusmaci: {meta['n_speakers']}")
        print(f"  Seed: {meta['seed']}")
        print(f"\n  {'Set':<10} {'Konusmaci':>12} {'Dosya':>10} {'Oran':>8}")
        print(f"  {'-' * 42}")
        total = meta["n_train_files"] + meta["n_val_files"] + meta["n_test_files"]
        for name in ["train", "val", "test"]:
            n_spk = meta[f"n_{name}_speakers"]
            n_files = meta[f"n_{name}_files"]
            ratio = n_files / total * 100 if total > 0 else 0
            print(f"  {name:<10} {n_spk:>12} {n_files:>10} {ratio:>7.1f}%")
        print("=" * 60)

    def save_splits(self, output_path: str):
        """Bolunme sonuclarini JSON dosyasina kaydeder."""
        splits = self.split()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)
        logger.info(f"Bolunme kaydedildi: {output_path}")
        return splits


def compute_generalization_gap(
    wer_seen: float, wer_unseen: float
) -> Dict:
    """Generalization gap hesaplar.

    Args:
        wer_seen: Egitimde gorulen konusmacilardaki WER
        wer_unseen: Ilk kez test edilen konusmacilardaki WER

    Returns:
        {"wer_seen": ..., "wer_unseen": ..., "delta_wer": ..., "generalization_ratio": ...}
    """
    delta = wer_unseen - wer_seen
    ratio = wer_unseen / wer_seen if wer_seen > 0 else float("inf")

    return {
        "wer_seen": round(wer_seen, 2),
        "wer_unseen": round(wer_unseen, 2),
        "delta_wer": round(delta, 2),
        "generalization_ratio": round(ratio, 3),
        "interpretation": (
            "Iyi genelleme" if delta < 5.0
            else "Orta genelleme" if delta < 15.0
            else "Zayif genelleme — konusmaci bagimli"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Konusmaci-Bagimsiz Veri Bolunmesi"
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--labels", default="data/processed/labels.json")
    parser.add_argument("--output", default="results/speaker_splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    splitter = SpeakerSplitter(
        data_dir=args.data_dir,
        labels_path=args.labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    splitter.save_splits(args.output)


if __name__ == "__main__":
    main()
