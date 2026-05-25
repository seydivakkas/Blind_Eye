"""
tools/update_labels.py
──────────────────────
data/processed/ altındaki tüm NPY klasörlerini tarar,
labels.json'a eksik eşleşmeleri otomatik ekler.

Kullanım:
    python tools/update_labels.py                     # Sadece rapor
    python tools/update_labels.py --apply             # Değişiklikleri uygula
    python tools/update_labels.py --apply --backup    # Yedek al + uygula
"""

import os
import sys
import json
import glob
import shutil
import argparse
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Türkçe karakter düzeltme haritası (dosya adı → gerçek kelime)
WORD_MAP = {
    "afiyetolsun": "afiyetolsun",
    "basla": "başla",
    "bitir": "bitir",
    "durdur": "durdur",
    "evet": "evet",
    "gorusmekuzere": "görüşmek üzere",
    "gunaydin": "günaydın",
    "hayir": "hayır",
    "hosgeldiniz": "hoşgeldiniz",
    "lutfen": "lütfen",
    "merhaba": "merhaba",
    "ozurdilerim": "özür dilerim",
    "selam": "selam",
    "tamam": "tamam",
    "tesekkurederim": "teşekkür ederim",
    "tesekkurler": "teşekkürler",
}

# CTC eğitimi için ASCII-uyumlu etiketler (Türkçe karakter destekli charset ile)
LABEL_MAP = {
    "afiyetolsun": "afiyetolsun",
    "basla": "basla",          # başla → charset'te ş var
    "bitir": "bitir",
    "durdur": "durdur",
    "evet": "evet",
    "gorusmekuzere": "gorusmekuzere",
    "gunaydin": "gunaydin",
    "hayir": "hayir",
    "hosgeldiniz": "hosgeldiniz",
    "lutfen": "lutfen",
    "merhaba": "merhaba",
    "ozurdilerim": "ozurdilerim",
    "selam": "selam",
    "tamam": "tamam",
    "tesekkurederim": "tesekkurederim",
    "tesekkurler": "tesekkurler",
}


def scan_processed_dir(data_root: str) -> dict:
    """data/processed/ altındaki tüm NPY dosyalarını tarar.

    Returns:
        dict: {rel_path: word_label} formatında eşleşmeler
    """
    all_entries = {}

    for class_dir in sorted(os.listdir(data_root)):
        class_path = os.path.join(data_root, class_dir)
        if not os.path.isdir(class_path) or class_dir == "__pycache__":
            continue

        # labels.json'daki formata uygun label belirle
        label = LABEL_MAP.get(class_dir, class_dir)

        npy_files = sorted(glob.glob(os.path.join(class_path, "*.npy")))
        for npy_path in npy_files:
            rel_path = f"{class_dir}/{os.path.basename(npy_path)}"
            all_entries[rel_path] = label

    return all_entries


def main():
    parser = argparse.ArgumentParser(description="labels.json Eksik Etiket Güncelleyici")
    parser.add_argument("--data-root", default="data/processed",
                        help="İşlenmiş veri dizini")
    parser.add_argument("--labels", default="data/processed/labels.json",
                        help="Mevcut labels.json yolu")
    parser.add_argument("--apply", action="store_true",
                        help="Değişiklikleri labels.json'a yaz")
    parser.add_argument("--backup", action="store_true",
                        help="Uygulama öncesi yedek al")
    args = parser.parse_args()

    # ── 1. Mevcut labels.json'u yükle ──
    if os.path.exists(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"Mevcut labels.json: {len(existing)} kayıt")
    else:
        existing = {}
        logger.warning(f"labels.json bulunamadı: {args.labels}")

    # ── 2. Dosya sistemini tara ──
    scanned = scan_processed_dir(args.data_root)
    logger.info(f"Taranan NPY dosyaları: {len(scanned)}")

    # ── 3. Eksik eşleşmeleri bul ──
    missing = {}
    for rel_path, label in scanned.items():
        if rel_path not in existing:
            missing[rel_path] = label

    # ── 4. Rapor ──
    logger.info(f"\n{'='*60}")
    logger.info(f"RAPOR")
    logger.info(f"{'='*60}")
    logger.info(f"  Mevcut labels.json kayıtları: {len(existing)}")
    logger.info(f"  Taranan NPY dosyaları:        {len(scanned)}")
    logger.info(f"  Eksik eşleşmeler:             {len(missing)}")

    # Kelime dağılımı (güncellenmiş)
    merged = {**existing, **missing}
    word_counts = Counter(merged.values())
    logger.info(f"\n  Güncellenmiş kelime dağılımı ({len(word_counts)} sınıf):")
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        status = " ← YENİ" if word in [LABEL_MAP[k] for k in LABEL_MAP
                                         if k in [os.path.dirname(p) for p in missing]] else ""
        logger.info(f"    {word:20s}: {count:4d}{status}")

    if not missing:
        logger.info("\n✅ Tüm NPY dosyaları zaten labels.json'da eşleştirilmiş.")
        return

    # Eksik eşleşmelerin sınıf dağılımı
    missing_classes = Counter(os.path.dirname(p) for p in missing)
    logger.info(f"\n  Eksik sınıflar:")
    for cls, cnt in sorted(missing_classes.items(), key=lambda x: -x[1]):
        logger.info(f"    {cls}: +{cnt} dosya")

    # ── 5. Uygula (opsiyonel) ──
    if args.apply:
        if args.backup and os.path.exists(args.labels):
            backup_path = args.labels + ".bak"
            shutil.copy2(args.labels, backup_path)
            logger.info(f"\n📋 Yedek alındı: {backup_path}")

        merged_sorted = dict(sorted(merged.items()))
        with open(args.labels, "w", encoding="utf-8") as f:
            json.dump(merged_sorted, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ labels.json güncellendi: {len(merged_sorted)} kayıt")
        logger.info(f"   Eklenen: {len(missing)} yeni eşleşme")
    else:
        logger.info(f"\n⚠️  Değişiklikler uygulanmadı. --apply flag'i ile çalıştırın:")
        logger.info(f"   python tools/update_labels.py --apply --backup")


if __name__ == "__main__":
    main()
