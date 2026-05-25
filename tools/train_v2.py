"""
tools/train_v2.py
─────────────────
V2 Model Egitim Pipeline — Cross-Lingual Transfer Learning destekli.

Ozellikler:
    - ResNet-18 + Conformer (LipReadModelV2)
    - Progressive unfreezing (frontend freeze -> unfreeze)
    - Curriculum learning (kisa -> uzun kelimeler)
    - Label smoothing CTC
    - Gelismis augmentasyon
    - ImageNet / LRW pretrained transfer

Kullanim:
    python tools/train_v2.py --epochs 100 --pretrained resnet18-imagenet --strategy progressive
    python tools/train_v2.py --epochs 50 --strategy finetune --resume models/checkpoints/v2_best.pth
"""

import os
import sys
import json
import time
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("train_v2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class LipReadDataset(Dataset):
    """NPY tabanlı lip reading dataset."""

    def __init__(self, root_dir, labels_path, max_frames=30, augment=False):
        self.root = root_dir
        self.max_frames = max_frames
        self.augment = augment
        self.samples = []

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        for rel_path, word in labels.items():
            full = os.path.join(root_dir, rel_path)
            if os.path.exists(full):
                self.samples.append((full, word))

        log.info(f"Dataset: {len(self.samples)} ornek yuklendi ({root_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, word = self.samples[idx]
        frames = np.load(path).astype(np.float32)

        # Normalize
        if frames.max() > 1.0:
            frames = frames / 255.0

        # Pad/truncate
        T = frames.shape[0]
        if T > self.max_frames:
            frames = frames[: self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((self.max_frames - T, *frames.shape[1:]), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)

        # Augmentasyon
        if self.augment:
            frames = self._augment(frames)

        # Ensure [T, H, W, C]
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]

        return torch.from_numpy(frames), word, min(T, self.max_frames)

    def _augment(self, frames):
        """Gelismis augmentasyon."""
        T, H, W = frames.shape[:3]
        C = frames.shape[3] if frames.ndim == 4 else 1
        f = frames.reshape(T, H, W) if C == 1 else frames

        # 1. Horizontal flip (%50)
        if random.random() < 0.5:
            f = f[:, :, ::-1].copy()

        # 2. TimeMask (rastgele 1-3 frame siyahla)
        if random.random() < 0.4:
            mask_len = random.randint(1, min(3, T - 1))
            start = random.randint(0, T - mask_len)
            f[start: start + mask_len] = 0.0

        # 3. CutOut (rastgele dikdortgen maskeleme)
        if random.random() < 0.3:
            ch, cw = random.randint(8, 24), random.randint(8, 24)
            y, x = random.randint(0, H - ch), random.randint(0, W - cw)
            f[:, y: y + ch, x: x + cw] = 0.0

        # 4. Brightness jitter
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            f = np.clip(f * factor, 0, 1)

        return f.reshape(T, H, W, 1) if C == 1 else f


def collate_fn(batch):
    """Custom collate: word -> CTC label indices."""
    vocab_path = os.path.join("configs", "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    charset = vocab["charset"]
    char2idx = {c: i for i, c in enumerate(charset)}

    clips, labels, clip_lens, label_lens = [], [], [], []
    for frames, word, frame_len in batch:
        clips.append(frames)
        clip_lens.append(frame_len)

        label = [char2idx.get(ch, 0) for ch in word if ch in char2idx]
        labels.extend(label)
        label_lens.append(len(label))

    clips = torch.stack(clips)
    labels = torch.tensor(labels, dtype=torch.long)
    clip_lens = torch.tensor(clip_lens, dtype=torch.long)
    label_lens = torch.tensor(label_lens, dtype=torch.long)

    return clips, labels, clip_lens, label_lens


# ═══════════════════════════════════════════════════════════════
#  Curriculum Learning
# ═══════════════════════════════════════════════════════════════

def sort_by_difficulty(dataset: LipReadDataset) -> list:
    """Kisa kelimeler (kolay) -> uzun kelimeler (zor) sirala."""
    indexed = [(i, len(dataset.samples[i][1])) for i in range(len(dataset))]
    indexed.sort(key=lambda x: x[1])
    return [i for i, _ in indexed]


# ═══════════════════════════════════════════════════════════════
#  WER/CER Metrikleri
# ═══════════════════════════════════════════════════════════════

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return dp[n][m]


def compute_wer_cer(predictions, targets):
    total_wer, total_cer, total_words, total_chars = 0, 0, 0, 0
    for pred, tgt in zip(predictions, targets):
        total_wer += levenshtein(pred.split(), tgt.split())
        total_words += max(len(tgt.split()), 1)
        total_cer += levenshtein(pred, tgt)
        total_chars += max(len(tgt), 1)
    wer = total_wer / max(total_words, 1)
    cer = total_cer / max(total_chars, 1)
    return wer, cer


def greedy_decode(logits, charset):
    """CTC greedy decode."""
    preds = logits.argmax(dim=-1)
    results = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                if idx < len(charset):
                    chars.append(charset[idx])
            prev = idx
        results.append("".join(chars))
    return results


# ═══════════════════════════════════════════════════════════════
#  Egitim Dongusu
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, ctc_loss, device, grad_clip=5.0):
    model.train()
    total_loss = 0
    for clips, labels, clip_lens, label_lens in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        clip_lens = clip_lens.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        loss = ctc_loss(log_probs, labels, clip_lens, label_lens)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, ctc_loss, charset, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for clips, labels, clip_lens, label_lens in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        clip_lens = clip_lens.to(device)
        label_lens = label_lens.to(device)

        logits = model(clips)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
        loss = ctc_loss(log_probs, labels, clip_lens, label_lens)
        total_loss += loss.item()

        # Decode
        preds = greedy_decode(logits, charset)
        all_preds.extend(preds)

        # Reconstruct targets
        offset = 0
        for ll in label_lens:
            ll = ll.item()
            target_indices = labels[offset: offset + ll]
            target_str = "".join(charset[i] for i in target_indices.cpu().tolist() if i < len(charset))
            all_targets.append(target_str)
            offset += ll

    avg_loss = total_loss / max(len(loader), 1)
    wer, cer = compute_wer_cer(all_preds, all_targets)
    return avg_loss, wer, cer, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="V2 Egitim Pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-conf-layers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--strategy", choices=["freeze", "finetune", "progressive"],
                        default="progressive")
    parser.add_argument("--unfreeze-epoch", type=int, default=10,
                        help="Progressive: frontend unfreeze epoch")
    parser.add_argument("--pretrained", type=str, default="resnet18-imagenet",
                        help="Pretrained model adi veya checkpoint yolu")
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum learning (kisa->uzun)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Vocab ─────────────────────────────────────────────────
    with open("configs/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    charset = vocab["charset"]
    num_classes = vocab["num_classes"]
    log.info(f"Vocab: {num_classes} sinif")

    # ── Dataset ───────────────────────────────────────────────
    labels_path = os.path.join(args.data_dir, "labels.json")
    dataset = LipReadDataset(args.data_dir, labels_path, args.max_frames, augment=True)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    log.info(f"Train: {train_size} | Val: {val_size}")

    # ── Model ─────────────────────────────────────────────────
    from backend.cbam import LipReadModelV2

    model = LipReadModelV2(
        num_classes=num_classes,
        d_model=args.d_model,
        n_conf_layers=args.n_conf_layers,
        pretrained_resnet=(args.pretrained == "resnet18-imagenet"),
    )

    # Transfer weights (eger custom checkpoint verilmisse)
    if args.pretrained and args.pretrained != "resnet18-imagenet":
        if os.path.exists(args.pretrained):
            from tools.transfer_weights import load_pretrained_weights, transfer_resnet18_to_v2
            ckpt = load_pretrained_weights(args.pretrained)
            transfer_resnet18_to_v2(ckpt, model)

    model = model.to(device)
    counts = model.param_count()
    log.info(f"Model parametreleri: {counts['total']:,}")
    for k, v in counts.items():
        if k != "total":
            log.info(f"  {k}: {v:,}")

    # ── Strategy ──────────────────────────────────────────────
    if args.strategy in ("freeze", "progressive"):
        model.freeze_frontend()
        log.info(f"Strateji: {args.strategy}")
        if args.strategy == "progressive":
            log.info(f"  Frontend unfreeze: epoch {args.unfreeze_epoch}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        log.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Training Loop ─────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []
    ckpt_dir = "models/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"V2 EGITIM BASLIYOR")
    log.info(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info(f"Strategy: {args.strategy}  |  d_model: {args.d_model}")
    log.info(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Progressive unfreezing
        if args.strategy == "progressive" and epoch == args.unfreeze_epoch:
            log.info(f"\n>>> PROGRESSIVE UNFREEZE (epoch {epoch}) <<<")
            param_groups = model.unfreeze_frontend(lr_scale=0.1)
            # Optimizer'i yeniden olustur
            optimizer = torch.optim.AdamW([
                {"params": model.frontend.parameters(), "lr": args.lr * 0.1},
                {"params": model.cbam.parameters(), "lr": args.lr},
                {"params": model.proj.parameters(), "lr": args.lr},
                {"params": model.conformer.parameters(), "lr": args.lr},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ctc_loss, device, args.grad_clip
        )

        # Validate
        val_loss, wer, cer, preds, targets = validate(
            model, val_loader, ctc_loss, charset, device
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        log.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"WER={wer*100:.1f}% | CER={cer*100:.1f}% | "
            f"LR={lr:.2e} | {elapsed:.1f}s"
        )

        # Ornek tahminler (her 5 epoch)
        if (epoch + 1) % 5 == 0 and preds:
            log.info("  Ornek tahminler:")
            for p, t in zip(preds[:3], targets[:3]):
                log.info(f"    Tahmin: '{p}'  Gercek: '{t}'")

        # Checkpoint
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wer": wer,
            "cer": cer,
            "lr": lr,
            "elapsed": elapsed,
            "strategy": args.strategy,
        }
        training_log.append(entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "wer": wer,
                "cer": cer,
                "config": {
                    "num_classes": num_classes,
                    "d_model": args.d_model,
                    "n_conf_layers": args.n_conf_layers,
                },
            }, os.path.join(ckpt_dir, "v2_best.pth"))
            log.info(f"  -> Yeni en iyi! val_loss={val_loss:.4f} kaydedildi.")
        else:
            patience_counter += 1

        # Last checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "val_loss": val_loss,
        }, os.path.join(ckpt_dir, "v2_last.pth"))

        # Training log
        with open("results/training_log_v2.json", "w") as f:
            json.dump(training_log, f, indent=2)

        # Early stopping
        if patience_counter >= args.patience:
            log.info(f"\nEarly stopping: {args.patience} epoch boyunca iyilesme yok.")
            break

    # ── Ozet ──────────────────────────────────────────────────
    best_entry = min(training_log, key=lambda x: x["val_loss"])
    log.info(f"\n{'='*60}")
    log.info(f"V2 EGITIM TAMAMLANDI")
    log.info(f"En iyi epoch  : {best_entry['epoch']}")
    log.info(f"En iyi val_loss: {best_entry['val_loss']:.4f}")
    log.info(f"WER           : {best_entry['wer']*100:.1f}%")
    log.info(f"CER           : {best_entry['cer']*100:.1f}%")
    log.info(f"Strateji      : {args.strategy}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
