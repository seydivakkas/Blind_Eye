"""
tools/train_model.py
────────────────────
Blind Eye — LipReadModelWithCBAM CTC Eğitimi

Kullanım:
    python tools/train_model.py
    python tools/train_model.py --epochs 50 --batch-size 8 --lr 1e-3
    python tools/train_model.py --resume models/checkpoints/best.pth
    python tools/train_model.py --use-augmented   # augmented veri ile eğit

Çıktı:
    models/checkpoints/best.pth        — En iyi val_loss checkpoint
    models/checkpoints/last.pth        — Son epoch checkpoint
    models/student_fp32.onnx           — ONNX export (eğitim sonunda)
    results/training_log.json          — Epoch bazlı metrikler
"""

import os
import sys
import json
import time
import argparse
import logging
import random
import math

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Torch import kontrolü ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_OK = True
    log.info(f"PyTorch {torch.__version__} yüklü. CUDA: {torch.cuda.is_available()}")
except ImportError:
    TORCH_OK = False
    log.error("PyTorch bulunamadı! Lütfen: pip install torch")
    sys.exit(1)

# ── Yerel modüller ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.cbam import LipReadModelWithCBAM


# ── Vocab yükle ────────────────────────────────────────────────────────────────
def load_vocab(path="configs/vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Dataset ────────────────────────────────────────────────────────────────────
class LipReadDataset(Dataset):
    """NPY chunk'ları + labels.json'dan oluşan dataset.

    Shape: [T, H, W, C] → float32 (normalize edilmiş, [0,1])
    Label: kelime string → karakter indis listesi
    """

    def __init__(self, data_root: str, labels_json: str, charset: list, max_frames: int = 30):
        self.data_root = data_root
        self.charset = charset
        self.char2idx = {c: i for i, c in enumerate(charset)}
        self.max_frames = max_frames

        with open(labels_json, "r", encoding="utf-8") as f:
            labels_map = json.load(f)

        self.samples = []
        missing = 0
        for rel_path, word in labels_map.items():
            npy_path = os.path.join(data_root, rel_path.replace("/", os.sep))
            if not os.path.exists(npy_path):
                missing += 1
                continue
            # Kelimeyi karakter indis listesine çevir
            char_ids = [self.char2idx.get(c, -1) for c in word.lower()]
            char_ids = [i for i in char_ids if i >= 0]  # bilinmeyen karakterler atla
            if char_ids:
                self.samples.append((npy_path, char_ids))

        log.info(f"Dataset: {len(self.samples)} örnek yüklendi ({missing} eksik atlandı)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, char_ids = self.samples[idx]
        clip = np.load(npy_path).astype(np.float32)  # [T, H, W, 1]

        # T normalizasyonu
        T = clip.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, *clip.shape[1:]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
        else:
            clip = clip[:self.max_frames]

        clip_tensor = torch.from_numpy(clip)          # [T, H, W, 1]
        label_tensor = torch.tensor(char_ids, dtype=torch.long)
        return clip_tensor, label_tensor


def collate_fn(batch):
    """Değişken uzunluklu label'ları pad'le, CTC için hazırla."""
    clips, labels = zip(*batch)
    clips_stacked = torch.stack(clips, dim=0)           # [B, T, H, W, 1]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)                       # CTC: tek 1D tensor

    return clips_stacked, labels_cat, label_lengths


# ── WER hesaplama ─────────────────────────────────────────────────────────────
def levenshtein(a, b):
    """Levenshtein mesafesi (edit distance)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def ctc_greedy_decode(logits, charset, blank_idx=0):
    """[T, num_classes] logits → string"""
    ids = logits.argmax(dim=-1).tolist()
    # blank ve tekrar kaldır
    decoded = []
    prev = None
    for i in ids:
        if i != blank_idx and i != prev:
            if i < len(charset):
                decoded.append(charset[i])
        prev = i
    return "".join(decoded)


def compute_wer_cer(preds, targets):
    """Batch WER ve CER hesapla."""
    total_wer_num, total_wer_den = 0, 0
    total_cer_num, total_cer_den = 0, 0

    for pred, ref in zip(preds, targets):
        # CER (karakter seviyesi)
        total_cer_num += levenshtein(list(pred), list(ref))
        total_cer_den += max(len(ref), 1)

        # WER (kelime = tüm string bir kelime olarak)
        pred_words = pred.split()
        ref_words = ref.split()
        total_wer_num += levenshtein(pred_words, ref_words)
        total_wer_den += max(len(ref_words), 1)

    cer = total_cer_num / max(total_cer_den, 1)
    wer = total_wer_num / max(total_wer_den, 1)
    return wer, cer


# ── Checkpoint kaydet/yükle ───────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if optimizer and "optimizer_state" in ck:
        optimizer.load_state_dict(ck["optimizer_state"])
    return ck.get("epoch", 0), ck.get("val_loss", float("inf"))


# ── ONNX Export ───────────────────────────────────────────────────────────────
def export_onnx(model, num_classes, chunk_size, out_path, device):
    """Eğitilmiş modeli ONNX'e export eder."""
    model.eval()
    dummy = torch.randn(1, chunk_size, 96, 96, 1).to(device)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "T"}, "logits": {0: "batch", 1: "T"}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    log.info(f"ONNX kaydedildi: {out_path} ({size_mb:.1f} MB)")


# ── Eğitim döngüsü ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, ctc_loss, device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for clips, labels_cat, label_lengths in loader:
        clips = clips.to(device)           # [B, T, H, W, 1]
        labels_cat = labels_cat.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        logits = model(clips)              # [B, T, num_classes]
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C] for CTC

        B, T = logits.size(0), logits.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss(log_probs, labels_cat, input_lengths, label_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("NaN/Inf loss, batch atlanıyor")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, ctc_loss, charset, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_refs = []

    for clips, labels_cat, label_lengths in loader:
        clips = clips.to(device)
        labels_cat = labels_cat.to(device)
        label_lengths = label_lengths.to(device)

        logits = model(clips)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
        B, T = logits.size(0), logits.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss(log_probs, labels_cat, input_lengths, label_lengths)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            n_batches += 1

        # Decode
        logits_cpu = logits.cpu()
        offset = 0
        for i in range(B):
            pred = ctc_greedy_decode(logits_cpu[i], charset)
            llen = label_lengths[i].item()
            ref_ids = labels_cat[offset: offset + llen].cpu().tolist()
            ref = "".join(charset[j] for j in ref_ids if j < len(charset))
            all_preds.append(pred)
            all_refs.append(ref)
            offset += llen

    val_loss = total_loss / max(n_batches, 1)
    wer, cer = compute_wer_cer(all_preds, all_refs)
    return val_loss, wer, cer, all_preds[:5], all_refs[:5]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Blind Eye Model Eğitimi")
    parser.add_argument("--data-root",     default="data/processed")
    parser.add_argument("--use-augmented", action="store_true",
                        help="data/augmented klasörünü de kullan")
    parser.add_argument("--vocab",         default="configs/vocab.json")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--max-frames",    type=int,   default=30)
    parser.add_argument("--val-split",     type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--resume",        default=None,
                        help="Checkpoint yolu (eğitimi devam ettir)")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--onnx-out",       default="models/student_fp32.onnx")
    parser.add_argument("--log-out",        default="results/training_log.json")
    parser.add_argument("--no-export",      action="store_true",
                        help="Eğitim sonunda ONNX export yapma")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Vocab
    vocab = load_vocab(args.vocab)
    charset = vocab["charset"]
    num_classes = vocab["num_classes"]
    log.info(f"Vocab: {num_classes} sınıf, {len(charset)} karakter")

    # Dataset(lar)
    labels_json = os.path.join(args.data_root, "labels.json")
    dataset = LipReadDataset(args.data_root, labels_json, charset, args.max_frames)

    if args.use_augmented and os.path.exists("data/augmented"):
        aug_labels = os.path.join("data/augmented", "labels.json")
        if os.path.exists(aug_labels):
            aug_ds = LipReadDataset("data/augmented", aug_labels, charset, args.max_frames)
            from torch.utils.data import ConcatDataset
            dataset = ConcatDataset([dataset, aug_ds])
            log.info(f"Augmented veri eklendi → Toplam: {len(dataset)} örnek")

    # Train / Val split
    total = len(dataset)
    val_size = max(1, int(total * args.val_split))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    log.info(f"Eğitim: {train_size} | Doğrulama: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Model
    model = LipReadModelWithCBAM(num_classes=num_classes, hidden_dim=args.hidden_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parametreleri: {total_params:,}")

    # Optimizer + Scheduler + Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume)
        log.info(f"Checkpoint yüklendi: epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Eğitim kaydı
    os.makedirs("results", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    training_log = []

    log.info("\n" + "="*60)
    log.info("EĞİTİM BAŞLIYOR")
    log.info(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info("="*60 + "\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device, scheduler)
        val_loss, wer, cer, sample_preds, sample_refs = eval_epoch(
            model, val_loader, ctc_loss, charset, device
        )

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"WER={wer*100:.1f}% | CER={cer*100:.1f}% | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Örnek tahminler (her 5 epoch'ta bir)
        if epoch % 5 == 0 and sample_preds:
            log.info("  Örnek tahminler:")
            for p, r in zip(sample_preds[:3], sample_refs[:3]):
                log.info(f"    Tahmin: '{p}'  Gerçek: '{r}'")

        # Checkpoint kaydet
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "wer": wer, "cer": cer, "lr": lr_now,
        }
        training_log.append(record)

        # Her epoch son checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss,
                        os.path.join(args.checkpoint_dir, "last.pth"))

        # En iyi checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.checkpoint_dir, "best.pth"))
            log.info(f"  -> Yeni en iyi! val_loss={val_loss:.4f} kaydedildi.")

        # Log dosyasına yaz
        with open(args.log_out, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    # ── Eğitim sonu özeti ─────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("EĞİTİM TAMAMLANDI")
    if training_log:
        best = min(training_log, key=lambda x: x["val_loss"])
        log.info(f"En iyi epoch  : {best['epoch']}")
        log.info(f"En iyi val_loss: {best['val_loss']:.4f}")
        log.info(f"WER           : {best['wer']*100:.1f}%")
        log.info(f"CER           : {best['cer']*100:.1f}%")
    log.info(f"Log           : {args.log_out}")
    log.info(f"Best checkpoint: {args.checkpoint_dir}/best.pth")
    log.info("="*60 + "\n")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    if not args.no_export:
        log.info("En iyi modeli yükleyip ONNX export yapılıyor...")
        best_ck = os.path.join(args.checkpoint_dir, "best.pth")
        if os.path.exists(best_ck):
            load_checkpoint(model, None, best_ck)
        export_onnx(model, num_classes, args.max_frames, args.onnx_out, device)
        log.info(f"Sonraki adım: python tools/export_to_onnx.py  (INT8 quantization için)")


if __name__ == "__main__":
    main()
