"""
tools/train_pi_real.py
───────────────────────
Raspberry Pi Zero uyumlu ultra-hafif Türkçe dudak okuma modelinin
gerçek Mendeley Türkçe Dudak Okuma Veriseti ile eğitilmesi ve ONNX formatına aktarılması.
"""

import os
import sys
import json
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections import Counter

# Logging ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cihaz seçimi (GPU varsa otomatik kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════
#  Lightweight Model Mimarisi (Pi Zero İçin Optimize)
# ═══════════════════════════════════════════════════════════════

class DepthwiseSeparableConv(nn.Module):
    """Raspberry Pi için optimize edilmiş derinlik bazlı ayrıştırılabilir konvolüsyon."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class MobileNetV3TinySpatial(nn.Module):
    """
    Raspberry Pi Zero'nun 512MB RAM ve tek çekirdek kısıtına göre tasarlanmış 
    özel ultra-hafif uzamsal (spatial) özellik çıkarıcı. (Giriş: [B, 1, 96, 96])
    """
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            # İlk katman: Standart Conv
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            
            # Derinlik bazlı bloklar
            DepthwiseSeparableConv(8, 16, stride=2),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            
            # Global Average Pooling ile boyut azaltma
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)  # [B, 64, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 64]
        return self.fc(features)  # [B, feature_dim]

class PiLipReadingModel(nn.Module):
    """
    Raspberry Pi Zero uyumlu uçtan uca uzamsal-zamansal (spatial-temporal) dudak okuma modeli.
    """
    def __init__(self, feature_dim: int = 64, num_classes: int = 31):
        super().__init__()
        self.spatial = MobileNetV3TinySpatial(feature_dim)
        
        # Zaman boyutu için 1D CNN CTC sınıflandırıcı (ağır RNN/Conformer bloklarından kaçınılmıştır)
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(feature_dim, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x boyutu: [B, T, 1, 96, 96]
        b, t, c, h, w = x.size()
        
        # Batch ve Time boyutlarını birleştirerek uzamsal özellik çıkarımı yap
        x_flat = x.view(b * t, c, h, w)
        features_flat = self.spatial(x_flat)  # [B*T, feature_dim]
        
        # Boyutu zamansal seriye geri dönüştür
        features = features_flat.view(b, t, -1)  # [B, T, feature_dim]
        
        # 1D CNN için permutasyon: [B, feature_dim, T]
        features = features.permute(0, 2, 1)
        
        logits = self.temporal(features)  # [B, num_classes, T]
        
        # Çıktıyı [B, T, num_classes] formatına geri getir
        return logits.permute(0, 2, 1)


class PiDualHeadModel(nn.Module):
    """Dual-Head CTC Modeli: Karakter + Viseme eşzamanlı çıktı.

    Paylaşılan spatial backbone + iki ayrı temporal decoder:
    - Head 1 (char): Karakter seviyesi CTC (num_char_classes sınıf)
    - Head 2 (viseme): Viseme seviyesi CTC (num_viseme_classes sınıf)

    Multi-task loss: L = L_char + λ * L_viseme
    """
    def __init__(self, feature_dim: int = 64,
                 num_char_classes: int = 31,
                 num_viseme_classes: int = 12):
        super().__init__()
        self.spatial = MobileNetV3TinySpatial(feature_dim)

        # Head 1: Karakter CTC
        self.temporal_char = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(feature_dim, num_char_classes, kernel_size=1)
        )

        # Head 2: Viseme CTC
        self.temporal_viseme = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(feature_dim, num_viseme_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        b, t, c, h, w = x.size()
        x_flat = x.view(b * t, c, h, w)
        features_flat = self.spatial(x_flat)
        features = features_flat.view(b, t, -1).permute(0, 2, 1)

        char_logits = self.temporal_char(features).permute(0, 2, 1)    # [B, T, char_C]
        viseme_logits = self.temporal_viseme(features).permute(0, 2, 1) # [B, T, vis_C]
        return char_logits, viseme_logits

# ═══════════════════════════════════════════════════════════════
#  Dataset ve Veri Yükleme
# ═══════════════════════════════════════════════════════════════

class LipReadDataset(Dataset):
    """NPY chunk'ları + labels.json'dan oluşan veriseti.

    Augmentation:
        - Yatay çevirme (flip): Dudak simetrisi korunur
        - Zamansal çitleme (jitter): ±2 frame rastgele başlangıç
        - Gauss gürültü: σ=0.02 piksel değişkenliği
    """
    def __init__(self, data_root: str, labels_json: str, charset: list,
                 max_frames: int = 30, augment: bool = False,
                 viseme_labels_json: str = None, viseme_charset: list = None):
        self.data_root = data_root
        self.charset = charset
        self.char2idx = {c: i for i, c in enumerate(charset)}
        self.max_frames = max_frames
        self.augment = augment
        self.use_viseme = viseme_labels_json is not None and viseme_charset is not None
        self.viseme_charset = viseme_charset
        self.viseme2idx = {v: i for i, v in enumerate(viseme_charset)} if viseme_charset else {}
        self.samples = []

        # Viseme etiketleri yükle
        viseme_map = {}
        if self.use_viseme and os.path.exists(viseme_labels_json):
            with open(viseme_labels_json, "r", encoding="utf-8") as f:
                viseme_map = json.load(f)
            logger.info(f"Viseme etiketleri yüklendi: {len(viseme_map)} kayıt")

        with open(labels_json, "r", encoding="utf-8") as f:
            labels_map = json.load(f)

        missing = 0
        for rel_path, word in labels_map.items():
            npy_path = os.path.join(data_root, rel_path.replace("/", os.sep))
            if not os.path.exists(npy_path):
                missing += 1
                continue
            
            # Kelimeyi karakter indis listesine çevir
            char_ids = [self.char2idx.get(c, -1) for c in word.lower()]
            char_ids = [i for i in char_ids if i >= 0]

            # Viseme indis dizisi
            viseme_ids = []
            if self.use_viseme and rel_path in viseme_map:
                viseme_seq = viseme_map[rel_path].split()
                viseme_ids = [self.viseme2idx.get(v, -1) for v in viseme_seq]
                viseme_ids = [i for i in viseme_ids if i >= 0]

            if char_ids:
                self.samples.append((npy_path, char_ids, word, viseme_ids))

        logger.info(f"Dataset: {len(self.samples)} örnek yüklendi ({missing} eksik atlandı)")

    def get_class_weights(self) -> list:
        """Sınıf dengesizliği için örnek ağırlıkları hesaplar.

        Az örnekli sınıflar yüksek ağırlık alır:
        weight_i = max_count / count_i
        """
        word_counts = Counter(s[2] for s in self.samples)
        max_count = max(word_counts.values())
        sample_weights = []
        for _, _, word, _ in self.samples:
            weight = max_count / word_counts[word]
            sample_weights.append(weight)
        return sample_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, char_ids, word, viseme_ids = self.samples[idx]
        clip = np.load(npy_path).astype(np.float32)  # [T, H, W, 1]

        # T normalizasyonu (Padding / Truncating)
        T = clip.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, *clip.shape[1:]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
        else:
            clip = clip[:self.max_frames]

        # ── Augmentation ──
        if self.augment:
            # Yatay çevirme (%50 olasılık)
            if random.random() > 0.5:
                clip = clip[:, :, ::-1, :].copy()

            # Zamansal çitleme (±2 frame shift)
            jitter = random.randint(-2, 2)
            if jitter != 0:
                clip = np.roll(clip, jitter, axis=0)
                if jitter > 0:
                    clip[:jitter] = 0.0
                else:
                    clip[jitter:] = 0.0

            # Gauss gürültü (σ=0.02)
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, clip.shape).astype(np.float32)
                clip = np.clip(clip + noise, 0.0, 1.0)

        clip_tensor = torch.from_numpy(clip)  # [T, H, W, 1]
        
        # [T, H, W, 1] -> [T, 1, H, W] (Model kanal boyutunu başta bekliyor)
        clip_tensor = clip_tensor.permute(0, 3, 1, 2)
        
        label_tensor = torch.tensor(char_ids, dtype=torch.long)
        viseme_tensor = torch.tensor(viseme_ids, dtype=torch.long) if viseme_ids else torch.tensor([], dtype=torch.long)
        return clip_tensor, label_tensor, word, viseme_tensor

def collate_fn(batch):
    """Değişken uzunluklu etiketleri birleştirir, CTC formatına hazırlar."""
    clips, labels, words, visemes = zip(*batch)
    clips_stacked = torch.stack(clips, dim=0)  # [B, T, 1, H, W]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)

    # Viseme labels
    viseme_lengths = torch.tensor([len(v) for v in visemes], dtype=torch.long)
    visemes_cat = torch.cat(visemes) if any(len(v) > 0 for v in visemes) else torch.tensor([], dtype=torch.long)

    return clips_stacked, labels_cat, label_lengths, words, visemes_cat, viseme_lengths

# ═══════════════════════════════════════════════════════════════
#  WER/CER Metrikleri ve Çözücü
# ═══════════════════════════════════════════════════════════════

def levenshtein(a, b):
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
    ids = logits.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for i in ids:
        if i != blank_idx and i != prev:
            if i < len(charset):
                decoded.append(charset[i])
        prev = i
    return "".join(decoded)

def compute_wer_cer(preds, targets):
    total_wer_num, total_wer_den = 0, 0
    total_cer_num, total_cer_den = 0, 0

    for pred, ref in zip(preds, targets):
        total_cer_num += levenshtein(list(pred), list(ref))
        total_cer_den += max(len(ref), 1)

        pred_words = pred.split()
        ref_words = ref.split()
        total_wer_num += levenshtein(pred_words, ref_words)
        total_wer_den += max(len(ref_words), 1)

    cer = total_cer_num / max(total_cer_den, 1)
    wer = total_wer_num / max(total_wer_den, 1)
    return wer, cer

# ═══════════════════════════════════════════════════════════════
#  Eğitim Döngüsü
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, ctc_loss, device,
                viseme_mode=False, viseme_ctc_loss=None, viseme_lambda=0.5):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for clips, labels_cat, label_lengths, _, visemes_cat, viseme_lengths in loader:
        clips = clips.to(device)
        labels_cat = labels_cat.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        if viseme_mode:
            char_logits, vis_logits = model(clips)
            # Karakter CTC loss
            char_lp = char_logits.log_softmax(dim=-1).permute(1, 0, 2)
            B, T = char_logits.size(0), char_logits.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss_char = ctc_loss(char_lp, labels_cat, input_lengths, label_lengths)

            # Viseme CTC loss
            loss_vis = torch.tensor(0.0, device=device)
            if visemes_cat.numel() > 0:
                visemes_cat = visemes_cat.to(device)
                viseme_lengths = viseme_lengths.to(device)
                vis_lp = vis_logits.log_softmax(dim=-1).permute(1, 0, 2)
                loss_vis = viseme_ctc_loss(vis_lp, visemes_cat, input_lengths, viseme_lengths)

            loss = loss_char + viseme_lambda * loss_vis
        else:
            logits = model(clips)  # [B, T, num_classes]
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            B, T = logits.size(0), logits.size(1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, labels_cat, input_lengths, label_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def eval_epoch(model, loader, ctc_loss, charset, device, viseme_mode=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_refs = []

    for clips, labels_cat, label_lengths, words, visemes_cat, viseme_lengths in loader:
        clips = clips.to(device)
        labels_cat = labels_cat.to(device)
        label_lengths = label_lengths.to(device)

        if viseme_mode:
            char_logits, _ = model(clips)
            logits = char_logits
        else:
            logits = model(clips)

        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
        B, T = logits.size(0), logits.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = ctc_loss(log_probs, labels_cat, input_lengths, label_lengths)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            n_batches += 1

        logits_cpu = logits.cpu()
        for i in range(B):
            pred = ctc_greedy_decode(logits_cpu[i], charset)
            ref = words[i]
            all_preds.append(pred)
            all_refs.append(ref)

    val_loss = total_loss / max(n_batches, 1)
    wer, cer = compute_wer_cer(all_preds, all_refs)
    return val_loss, wer, cer, all_preds[:5], all_refs[:5]

# ═══════════════════════════════════════════════════════════════
#  ONNX İhracı
# ═══════════════════════════════════════════════════════════════

def export_onnx(model, max_frames, out_path, device):
    model.eval()
    dummy = torch.randn(1, max_frames, 1, 96, 96).to(device)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 1: "T"},
            "logits": {0: "batch", 1: "T"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    logger.info(f"ONNX modeli başarıyla dışa aktarıldı: {out_path} ({size_mb:.3f} MB)")

# ═══════════════════════════════════════════════════════════════
#  Eğitim Eğrisi Kaydetme
# ═══════════════════════════════════════════════════════════════

def _save_training_curves(history: dict, epochs: int):
    """Eğitim eğrilerini PNG olarak kaydeder."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(range(1, epochs + 1), history["train_loss"], label="Train Loss")
        axes[0].plot(range(1, epochs + 1), history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CTC Loss")
        axes[0].set_title("Kayıp Fonksiyonu")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(1, epochs + 1), history["wer"], color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("WER (%)")
        axes[1].set_title("Kelime Hata Oranı (WER)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(range(1, epochs + 1), history["cer"], color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("CER (%)")
        axes[2].set_title("Karakter Hata Oranı (CER)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/training_curves.png", dpi=150)
        plt.close()
        logger.info("Eğitim eğrileri kaydedildi: results/training_curves.png")
    except ImportError:
        logger.warning("matplotlib yüklü değil — eğitim eğrileri atlandı.")

# ═══════════════════════════════════════════════════════════════
#  Ana Metot
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Pi Zero Türkçe Dudak Okuma Modeli Gerçek Eğitim Betiği")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--vocab", default="configs/vocab.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--no-augment", action="store_true", help="Veri augmentasyonunu devre dışı bırak")
    parser.add_argument("--no-weighted", action="store_true", help="Sınıf ağırlıklı örneklemeyi kapat")
    parser.add_argument("--onnx-out", default="models/pi_model_float32.onnx")
    parser.add_argument("--checkpoint-out", default="models/checkpoints/pi_model_best.pth")
    parser.add_argument("--export-only", action="store_true", help="Eğitimi atlayıp doğrudan ONNX'e ihraç et")
    parser.add_argument("--viseme", action="store_true", help="Dual-head CTC (karakter + viseme) modunu aktifleştir")
    parser.add_argument("--viseme-vocab", default="configs/viseme_vocab.json", help="Viseme vocab dosyası")
    parser.add_argument("--viseme-lambda", type=float, default=0.5, help="Viseme loss ağırlığı (λ)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    logger.info(f"Cihaz: {device}")

    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    charset = vocab["charset"]
    num_classes = vocab["num_classes"]
    logger.info(f"Vocab: {num_classes} sınıf yüklendi.")

    # Viseme modu
    viseme_charset = None
    num_viseme_classes = 0
    if args.viseme:
        with open(args.viseme_vocab, "r", encoding="utf-8") as f:
            vis_vocab = json.load(f)
        viseme_charset = vis_vocab["charset"]
        num_viseme_classes = vis_vocab["num_classes"]
        logger.info(f"Viseme modu aktif: {num_viseme_classes} viseme sınıfı")

    labels_json = os.path.join(args.data_root, "labels.json")
    viseme_labels_json = os.path.join(args.data_root, "viseme_labels.json") if args.viseme else None
    dataset = LipReadDataset(
        args.data_root, labels_json, charset, args.max_frames,
        augment=not args.no_augment,
        viseme_labels_json=viseme_labels_json,
        viseme_charset=viseme_charset
    )

    total = len(dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Eğitim Örnekleri: {train_size} | Doğrulama Örnekleri: {val_size}")

    # WeightedRandomSampler ile sınıf dengesizliği çözümü
    sampler = None
    shuffle_train = True
    if not args.no_weighted:
        all_weights = dataset.get_class_weights()
        train_weights = [all_weights[i] for i in train_ds.indices]
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights), replacement=True
        )
        shuffle_train = False
        logger.info("WeightedRandomSampler aktif (az örnekli sınıflar dengelendi)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle_train,
        sampler=sampler, collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    if args.viseme:
        model = PiDualHeadModel(
            feature_dim=64, num_char_classes=num_classes,
            num_viseme_classes=num_viseme_classes
        ).to(device)
        logger.info("PiDualHeadModel (Karakter + Viseme) yüklendi")
    else:
        model = PiLipReadingModel(feature_dim=64, num_classes=num_classes).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parametre sayısı: {params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    viseme_ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True) if args.viseme else None

    best_val_loss = float("inf")
    checkpoint = {}

    if not args.export_only:
        history = {"train_loss": [], "val_loss": [], "wer": [], "cer": []}
        logger.info(f"\n{'='*60}\nEĞİTİM BAŞLIYOR (Epoch: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr})\n{'='*60}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, ctc_loss, device,
                viseme_mode=args.viseme, viseme_ctc_loss=viseme_ctc_loss,
                viseme_lambda=args.viseme_lambda
            )
            val_loss, wer, cer, sample_preds, sample_refs = eval_epoch(
                model, val_loader, ctc_loss, charset, device,
                viseme_mode=args.viseme
            )
            scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["wer"].append(wer * 100)
            history["cer"].append(cer * 100)

            logger.info(
                f"Epoch {epoch:2d}/{args.epochs:2d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"WER={wer*100:.1f}% | CER={cer*100:.1f}% | "
                f"Süre={elapsed:.1f}sn"
            )

            if epoch % 2 == 0 and sample_preds:
                logger.info("  Örnek Tahminler:")
                for p, r in zip(sample_preds[:3], sample_refs[:3]):
                    logger.info(f"    Tahmin: '{p}' <-> Gerçek: '{r}'")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "wer": wer, "cer": cer
                }, args.checkpoint_out)
                logger.info(f"  -> Yeni en iyi model kaydedildi! val_loss={val_loss:.4f}")

        logger.info(f"\nEğitim başarıyla tamamlandı! En iyi val_loss={best_val_loss:.4f}\n")
        _save_training_curves(history, args.epochs)
    else:
        logger.info("Eğitim adımı atlandı (--export-only aktif).")

    if os.path.exists(args.checkpoint_out):
        logger.info("En iyi ağırlıklar yükleniyor...")
        checkpoint = torch.load(args.checkpoint_out, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        best_val_loss = checkpoint.get("val_loss", 0.0)
    else:
        logger.error(f"Checkpoint bulunamadı: {args.checkpoint_out}")
        sys.exit(1)

    logger.info("Model ONNX formatına aktarılıyor...")
    export_onnx(model, args.max_frames, args.onnx_out, device)

    metrics_path = "results/pi_train_metrics.json"
    os.makedirs("results", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "PiLipReadingModel (MobileNetV3-Tiny + 1D-CNN)",
            "params": params,
            "best_val_loss": best_val_loss,
            "best_wer": checkpoint.get("wer", 1.0) * 100,
            "best_cer": checkpoint.get("cer", 1.0) * 100,
            "size_mb": os.path.getsize(args.onnx_out) / (1024 ** 2)
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Eğitim metrikleri kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main()
