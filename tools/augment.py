"""
tools/augment.py
Lip Reading Veri Augmentasyonu — SOTA Teknikler (Auto-AVSR, 2024)

Time Masking, Mixup, Speed Perturbation, CutMix, Temporal Warping,
Channel Dropout, Flip, Brightness/Noise

Kullanım:
    python tools/augment.py --input data/processed --output data/augmented --factor 3

Her orijinal örneği N kez augmentasyonla çoğaltır.

Literatür:
    - Auto-AVSR (Meta, 2024): Temporal masking en etkili teknik
    - DC-TCN (2024): Mixup + speed perturbation kombine etki
"""

import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  1. AUGMENTASYON TEKNİKLERİ
# ═══════════════════════════════════════════════════════════════

def time_mask(clip: np.ndarray, max_mask_len: int = 5) -> np.ndarray:
    """Rastgele ardışık frame'leri sıfırlar (SpecAugment benzeri).

    En etkili lip reading augmentasyonu — modeli eksik temporal
    bilgiye karşı dayanıklı yapar.
    """
    result = clip.copy()
    T = clip.shape[0]
    mask_len = np.random.randint(1, min(max_mask_len + 1, T // 2))
    start = np.random.randint(0, T - mask_len)
    result[start:start + mask_len] = 0.0
    return result


def mixup(clip_a: np.ndarray, clip_b: np.ndarray,
          alpha: float = 0.3) -> Tuple[np.ndarray, float]:
    """İki clip'i karıştırır (Mixup regularization).

    Returns:
        (mixed_clip, lambda) — lambda ağırlık katsayısı
    """
    lam = np.random.beta(alpha, alpha)
    # Shape uyumu sağla
    min_t = min(clip_a.shape[0], clip_b.shape[0])
    mixed = lam * clip_a[:min_t] + (1 - lam) * clip_b[:min_t]
    return mixed.astype(np.float32), lam


def horizontal_flip(clip: np.ndarray) -> np.ndarray:
    """Yatay aynalama — dudak simetrisi korunur."""
    return np.flip(clip, axis=2).copy()


def random_brightness(clip: np.ndarray,
                      delta_range: Tuple[float, float] = (-0.15, 0.15)) -> np.ndarray:
    """Rastgele parlaklık değişimi."""
    delta = np.random.uniform(*delta_range)
    result = np.clip(clip + delta, 0.0, 1.0)
    return result.astype(np.float32)


def gaussian_noise(clip: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Gaussian gürültü ekleme — kamera gürültüsü simülasyonu."""
    noise = np.random.randn(*clip.shape).astype(np.float32) * sigma
    result = np.clip(clip + noise, 0.0, 1.0)
    return result.astype(np.float32)


def random_crop_resize(clip: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """Rastgele kırpma + resize — pozisyon değişimine dayanıklılık."""
    import cv2
    T, H, W, C = clip.shape
    new_h = int(H * crop_ratio)
    new_w = int(W * crop_ratio)
    y = np.random.randint(0, H - new_h + 1)
    x = np.random.randint(0, W - new_w + 1)

    result = np.zeros_like(clip)
    for t in range(T):
        cropped = clip[t, y:y + new_h, x:x + new_w, :]
        result[t] = cv2.resize(
            cropped.squeeze(), (W, H), interpolation=cv2.INTER_LINEAR
        )[..., np.newaxis]
    return result


def speed_perturbation(clip: np.ndarray,
                       factor_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Konuşma hızı değişimi — frame sayısını interpolasyonla değiştirir.

    Auto-AVSR (Meta, 2024): Temporal augmentation ailesinin en etkili
    ikinci tekniği. Gerçek konuşma hız varyasyonlarını simüle eder.

    Args:
        clip: [T, H, W, C] video clip
        factor_range: (min_factor, max_factor) — 0.9x yavaş, 1.1x hızlı
    """
    factor = np.random.uniform(*factor_range)
    T = clip.shape[0]
    new_T = max(2, int(T * factor))

    # Linear interpolasyon ile yeni frame sayısına ölçekle
    old_indices = np.linspace(0, T - 1, new_T)
    result = np.zeros((new_T, *clip.shape[1:]), dtype=np.float32)

    for i, idx in enumerate(old_indices):
        low = int(np.floor(idx))
        high = min(low + 1, T - 1)
        weight = idx - low
        result[i] = (1 - weight) * clip[low] + weight * clip[high]

    return result


def cutmix(clip_a: np.ndarray, clip_b: np.ndarray,
           alpha: float = 1.0) -> np.ndarray:
    """CutMix — frame bölgesi değiştirme.

    Mixup'tan daha agresif: dikdörtgen bir bölgeyi başka clip'ten alır.
    Occlusion'a ve kısmi bilgiye karşı dayanıklılık sağlar.
    """
    lam = np.random.beta(alpha, alpha)
    T = min(clip_a.shape[0], clip_b.shape[0])
    H, W = clip_a.shape[1], clip_a.shape[2]

    # Kare boyutu
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))

    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    result = clip_a[:T].copy()
    result[:, y1:y2, x1:x2, :] = clip_b[:T, y1:y2, x1:x2, :]
    return result


def temporal_warping(clip: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Non-linear temporal bozulma — gerçek konuşma hız varyasyonları.

    Monoton artan warp fonksiyonu ile frame sırasını non-linearly değiştirir.
    Linear speed perturbation'dan farklı olarak, bazı bölümler hızlanırken
    diğerleri yavaşlar.
    """
    T = clip.shape[0]
    if T < 4:
        return clip

    # Monoton artan warp path üret
    control_points = np.linspace(0, T - 1, 5)
    offsets = np.random.randn(5) * sigma * T / 5
    offsets[0] = 0  # Başlangıç sabit
    offsets[-1] = 0  # Son sabit
    warped_points = control_points + offsets

    # Monotonluk garantisi
    for i in range(1, len(warped_points)):
        warped_points[i] = max(warped_points[i], warped_points[i - 1] + 0.5)
    warped_points = np.clip(warped_points, 0, T - 1)

    # Tüm frame'ler için interpolasyon
    source_indices = np.interp(
        np.arange(T),
        np.linspace(0, T - 1, len(warped_points)),
        warped_points
    )

    result = np.zeros_like(clip)
    for i, idx in enumerate(source_indices):
        low = int(np.floor(idx))
        high = min(low + 1, T - 1)
        weight = idx - low
        result[i] = (1 - weight) * clip[low] + weight * clip[high]

    return result


def channel_dropout(clip: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Rastgele piksel/bölge dropout — occlusion simülasyonu.

    Dudak bölgesinin kısmen kapatılması durumunu simüle eder
    (parmak, bardak, maske vb.).
    """
    result = clip.copy()
    mask = np.random.random(clip.shape[:3]) > p  # [T, H, W]
    mask = mask[..., np.newaxis]  # [T, H, W, 1]
    result = result * mask
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  2. AUGMENTASYON PIPELINE
# ═══════════════════════════════════════════════════════════════

class AugmentationPipeline:
    """Rastgele augmentasyon kombinasyonları uygular."""

    def __init__(self, all_clips: list = None):
        """
        Args:
            all_clips: Mixup için tüm clip'lerin listesi (opsiyonel)
        """
        self.all_clips = all_clips or []

    def augment(self, clip: np.ndarray) -> np.ndarray:
        """Rastgele augmentasyon kombinasyonu uygular.

        Olasılıklar Auto-AVSR (Meta, 2024) ablasyon çalışmasına göre
        kalibre edilmiştir. Temporal augmentation'lar en yüksek etkiye sahip.
        """
        result = clip.copy()

        # ── Temporal Augmentation (En Yüksek Etki) ──
        # %70 olasılıkla Time Masking (en etkili — Meta 2024)
        if np.random.random() < 0.7:
            result = time_mask(result)

        # %35 olasılıkla Speed Perturbation (0.9x-1.1x)
        if np.random.random() < 0.35:
            result = speed_perturbation(result)

        # %20 olasılıkla Temporal Warping (non-linear)
        if np.random.random() < 0.20:
            result = temporal_warping(result)

        # ── Spatial Augmentation ──
        # %40 olasılıkla Horizontal Flip
        if np.random.random() < 0.4:
            result = horizontal_flip(result)

        # %50 olasılıkla Brightness
        if np.random.random() < 0.5:
            result = random_brightness(result)

        # %30 olasılıkla Gaussian Noise
        if np.random.random() < 0.3:
            result = gaussian_noise(result)

        # %20 olasılıkla Random Crop Resize
        if np.random.random() < 0.2:
            result = random_crop_resize(result)

        # %15 olasılıkla Channel Dropout (occlusion)
        if np.random.random() < 0.15:
            result = channel_dropout(result)

        # ── Mix Augmentation ──
        # %15 olasılıkla Mixup (eğer diğer clip'ler varsa)
        if np.random.random() < 0.15 and len(self.all_clips) > 1:
            other = self.all_clips[np.random.randint(len(self.all_clips))]
            result, _ = mixup(result, other)

        # %10 olasılıkla CutMix (daha agresif)
        if np.random.random() < 0.10 and len(self.all_clips) > 1:
            other = self.all_clips[np.random.randint(len(self.all_clips))]
            result = cutmix(result, other)

        return result


# ═══════════════════════════════════════════════════════════════
#  3. BATCH AUGMENTASYON
# ═══════════════════════════════════════════════════════════════

def augment_dataset(
    input_dir: str,
    output_dir: str,
    factor: int = 3,
) -> dict:
    """Veri setini augmentasyonla çoğaltır.

    Args:
        input_dir: preprocess_dataset.py çıktısı
        output_dir: augmentasyonlu veri çıktısı
        factor: Her örneği kaç kez çoğalt (orijinal + factor adet)

    Returns:
        stats dict
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    stats = {"original": 0, "augmented": 0, "total": 0}

    # Tüm .npy dosyalarını tara
    all_npy_files = sorted(input_path.rglob("*.npy"))
    logger.info(f"📂 {len(all_npy_files)} orijinal örnek bulundu.")

    # Mixup için tüm clip'leri yükle (bellek izin veriyorsa)
    all_clips = []
    if len(all_npy_files) < 5000:  # 5K'dan az ise RAM'e sığar
        for f in all_npy_files:
            try:
                all_clips.append(np.load(str(f)))
            except Exception:
                pass

    pipeline = AugmentationPipeline(all_clips=all_clips)

    # Labels yükle
    labels_path = input_path / "labels.json"
    labels = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    new_labels = {}

    for npy_file in all_npy_files:
        relative = npy_file.relative_to(input_path)
        sample_id = npy_file.stem
        label = labels.get(sample_id, relative.parent.name)

        # Orijinali kopyala
        out_dir = output_path / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        clip = np.load(str(npy_file))

        # Orijinal kaydet
        out_orig = out_dir / f"{sample_id}_orig.npy"
        np.save(str(out_orig), clip)
        new_labels[f"{sample_id}_orig"] = label
        stats["original"] += 1

        # Augmentasyonlu kopyalar
        for i in range(factor):
            aug_clip = pipeline.augment(clip)
            out_aug = out_dir / f"{sample_id}_aug{i:02d}.npy"
            np.save(str(out_aug), aug_clip)
            new_labels[f"{sample_id}_aug{i:02d}"] = label
            stats["augmented"] += 1

    stats["total"] = stats["original"] + stats["augmented"]

    # Labels kaydet
    out_labels = output_path / "labels.json"
    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)

    logger.info(f"\n📊 Augmentasyon Tamamlandı:")
    logger.info(f"  Orijinal:     {stats['original']}")
    logger.info(f"  Augmentasyon: {stats['augmented']}")
    logger.info(f"  Toplam:       {stats['total']} (x{factor + 1} çoğaltma)")

    return stats


# ═══════════════════════════════════════════════════════════════
#  4. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lip Reading Veri Augmentasyonu"
    )
    parser.add_argument("--input", default="data/processed",
                        help="preprocess_dataset.py çıktı klasörü")
    parser.add_argument("--output", default="data/augmented",
                        help="Augmentasyonlu çıktı klasörü")
    parser.add_argument("--factor", type=int, default=3,
                        help="Çoğaltma faktörü (3 = orijinal + 3 kopya)")

    args = parser.parse_args()
    augment_dataset(args.input, args.output, args.factor)
