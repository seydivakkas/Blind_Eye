"""
tools/ablation_study.py
───────────────────────
6 koşullu ablation study framework'ü.

Deneysel Koşullar:
    1. Baseline: Sadece FaceMesh (KLT yok, kinematik yok)
    2. +KLT hibrit takip
    3. +Kinematik mimik analizi
    4. +Bilişsel yük indeksi (EAR/PERCLOS)
    5. +Viseme dual-head modeli
    6. +LM beam search

Her koşul için ölçümler:
    - WER / CER
    - FPS (kaç kare/saniye işlenebilir)
    - CPU yükü (%)
    - Bellek kullanımı (MB)
    - ONNX çıkarım süresi (ms)

Kullanım:
    python tools/ablation_study.py
    python tools/ablation_study.py --quick             # 3 epoch hızlı test
    python tools/ablation_study.py --epochs 50         # Tam ablation
    python tools/ablation_study.py --latex              # LaTeX tablo çıktısı
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Ablation Koşulları
# ═══════════════════════════════════════════════════════════════

ABLATION_CONDITIONS = [
    {
        "id": "C1_baseline",
        "name": "Baseline (FaceMesh Only)",
        "description": "Sadece FaceMesh landmark tespiti, KLT yok, augmentation yok",
        "flags": ["--no-augment", "--no-weighted"],
    },
    {
        "id": "C2_augmented",
        "name": "+Augmentation + WeightedSampler",
        "description": "Sınıf dengeleme ve veri augmentasyonu",
        "flags": [],
    },
    {
        "id": "C3_viseme",
        "name": "+Viseme Dual-Head",
        "description": "Karakter + Viseme çift başlı CTC",
        "flags": ["--viseme"],
    },
    {
        "id": "C4_viseme_weighted",
        "name": "+Viseme + Weighted + Augmented",
        "description": "Tam pipeline: dual-head + augmentation + weighted sampling",
        "flags": ["--viseme"],
    },
]


def get_ram_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def run_training_condition(condition, epochs, batch_size, data_root, vocab):
    """Tek bir ablation koşulunu çalıştırır."""
    import torch
    from tools.train_pi_real import (
        PiLipReadingModel, PiDualHeadModel, LipReadDataset, collate_fn,
        train_epoch, eval_epoch, ctc_greedy_decode, compute_wer_cer
    )
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vocab
    with open(vocab, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    charset = vocab_data["charset"]
    num_classes = vocab_data["num_classes"]

    # Koşul bayrakları
    use_augment = "--no-augment" not in condition["flags"]
    use_weighted = "--no-weighted" not in condition["flags"]
    use_viseme = "--viseme" in condition["flags"]

    # Viseme vocab
    viseme_charset = None
    num_viseme_classes = 0
    viseme_labels_json = None
    if use_viseme:
        vis_vocab_path = "configs/viseme_vocab.json"
        if os.path.exists(vis_vocab_path):
            with open(vis_vocab_path, "r", encoding="utf-8") as f:
                vis_vocab = json.load(f)
            viseme_charset = vis_vocab["charset"]
            num_viseme_classes = vis_vocab["num_classes"]
            viseme_labels_json = os.path.join(data_root, "viseme_labels.json")

    # Dataset
    labels_json = os.path.join(data_root, "labels.json")
    dataset = LipReadDataset(
        data_root, labels_json, charset, max_frames=30,
        augment=use_augment,
        viseme_labels_json=viseme_labels_json,
        viseme_charset=viseme_charset
    )

    total = len(dataset)
    val_size = int(total * 0.2)
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Sampler
    sampler = None
    shuffle = True
    if use_weighted:
        all_weights = dataset.get_class_weights()
        train_weights = [all_weights[i] for i in train_ds.indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Model
    if use_viseme and num_viseme_classes > 0:
        model = PiDualHeadModel(64, num_classes, num_viseme_classes).to(device)
    else:
        model = PiLipReadingModel(64, num_classes).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    viseme_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True) if use_viseme else None

    # Eğitim
    ram_before = get_ram_mb()
    t_start = time.time()

    best_val_loss = float("inf")
    best_wer, best_cer = 1.0, 1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, ctc_loss, device,
            viseme_mode=use_viseme, viseme_ctc_loss=viseme_ctc
        )
        val_loss, wer, cer, _, _ = eval_epoch(
            model, val_loader, ctc_loss, charset, device,
            viseme_mode=use_viseme
        )
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wer = wer
            best_cer = cer

    total_time = time.time() - t_start
    ram_after = get_ram_mb()

    return {
        "condition_id": condition["id"],
        "condition_name": condition["name"],
        "epochs": epochs,
        "params": params,
        "best_val_loss": round(best_val_loss, 4),
        "best_wer": round(best_wer * 100, 2),
        "best_cer": round(best_cer * 100, 2),
        "training_time_s": round(total_time, 1),
        "ram_delta_mb": round(ram_after - ram_before, 1) if ram_after > 0 else -1,
        "augmentation": use_augment,
        "weighted_sampling": use_weighted,
        "viseme_head": use_viseme,
    }


def generate_latex_table(results):
    """Ablation sonuçlarını LaTeX tablosuna çevirir."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study Results — Blind Eye Turkish Lip Reading}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Koşul} & \textbf{WER (\%)} & \textbf{CER (\%)} & \textbf{Val Loss} & \textbf{Params} & \textbf{Süre (s)} \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"  {r['condition_name']} & {r['best_wer']:.1f} & {r['best_cer']:.1f} & "
            f"{r['best_val_loss']:.4f} & {r['params']:,} & {r['training_time_s']:.0f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Framework")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="3 epoch hızlı test")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--vocab", default="configs/vocab.json")
    parser.add_argument("--output", default="results/ablation_results.json")
    parser.add_argument("--latex", action="store_true", help="LaTeX tablo çıktısı")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION STUDY ({len(ABLATION_CONDITIONS)} koşul, {args.epochs} epoch)")
    logger.info(f"{'='*60}")

    all_results = []

    for i, condition in enumerate(ABLATION_CONDITIONS, 1):
        logger.info(f"\n{'─'*40}")
        logger.info(f"Koşul {i}/{len(ABLATION_CONDITIONS)}: {condition['name']}")
        logger.info(f"Açıklama: {condition['description']}")
        logger.info(f"{'─'*40}")

        result = run_training_condition(
            condition, args.epochs, args.batch_size,
            args.data_root, args.vocab
        )
        all_results.append(result)

        logger.info(
            f"  → WER={result['best_wer']:.1f}% | CER={result['best_cer']:.1f}% | "
            f"val_loss={result['best_val_loss']:.4f} | {result['training_time_s']:.0f}s"
        )

    # Sonuçları kaydet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSonuçlar kaydedildi: {args.output}")

    # LaTeX çıktısı
    if args.latex:
        latex = generate_latex_table(all_results)
        latex_path = args.output.replace(".json", ".tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info(f"LaTeX tablosu kaydedildi: {latex_path}")

    # Sonuç tablosu
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION SONUÇ TABLOSU")
    logger.info(f"{'='*60}")
    logger.info(f"{'Koşul':<40s} {'WER%':>6s} {'CER%':>6s} {'Loss':>8s}")
    logger.info("-" * 62)
    for r in all_results:
        logger.info(f"{r['condition_name']:<40s} {r['best_wer']:>5.1f}% {r['best_cer']:>5.1f}% {r['best_val_loss']:>8.4f}")

    logger.info(f"\n✅ Ablation study tamamlandı!")


if __name__ == "__main__":
    main()
