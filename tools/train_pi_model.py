"""
tools/train_pi_model.py
Raspberry Pi Zero uyumlu, ultra-hafif Türkçe dudak okuma modeli eğitim ve dışa aktarım betiği.
"""

import os
import sys
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List

# Logging ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cihaz ayarı (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Karakter seti (Vocab)
VOCAB = list("<blank>") + list("abcçdefgğhıijklmnoöprsştuüvyz ")
NUM_CLASSES = len(VOCAB)

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
        # x boyutu: [B, 1, 96, 96]
        features = self.features(x)  # [B, 64, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 64]
        return self.fc(features)  # [B, feature_dim]

class PiLipReadingModel(nn.Module):
    """
    Raspberry Pi Zero uyumlu uçtan uca uzamsal-zamansal (spatial-temporal) dudak okuma modeli.
    """
    def __init__(self, feature_dim: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.spatial = MobileNetV3TinySpatial(feature_dim)
        
        # Zaman boyutu için 1D CNN CTC sınıflandırıcı (ağır RNN/Transformer bloklarından kaçınılmıştır)
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

def generate_synthetic_data(num_samples: int = 50, seq_len: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Demo eğitimi için sentetik dudak okuma veri kümesi ve CTC etiketleri üretir."""
    # Giriş görüntüsü: [N, T, 1, 96, 96]
    x = torch.randn(num_samples, seq_len, 1, 96, 96, dtype=torch.float32)
    
    # Hedef etiketler (1 ile 30 arasında rastgele token dizileri, blank=0 hariç)
    # CTC için her örnek farklı uzunlukta olabilir. Kolaylık olması için sabit 4 karakterli yapalım.
    targets = torch.randint(1, NUM_CLASSES, (num_samples, 4), dtype=torch.long)
    
    # Giriş ve hedef uzunluk vektörleri (CTC Loss için gerekli)
    input_lengths = torch.full((num_samples,), seq_len, dtype=torch.long)
    target_lengths = torch.full((num_samples,), 4, dtype=torch.long)
    
    return x, targets, input_lengths, target_lengths

def main():
    logger.info("Pi Zero Türkçe Dudak Okuma Modeli Eğitim Süreci Başlatıldı.")
    
    # 1. Sentetik Veri Kümesi Üretimi (Demo)
    logger.info("Sentetik eğitim veri kümesi oluşturuluyor...")
    x, y, input_lens, target_lens = generate_synthetic_data(num_samples=100, seq_len=6)
    
    # 2. Model ve Optimizer Kurulumu
    model = PiLipReadingModel(feature_dim=64, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # CTC Loss Fonksiyonu (Blank=0)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # 3. Eğitim Döngüsü
    logger.info("Model eğitimi başlatılıyor (5 Epoch)...")
    model.train()
    
    for epoch in range(1, 6):
        optimizer.zero_grad()
        
        # İleri yayılım (Forward pass)
        outputs = model(x.to(device))  # [B, T, num_classes]
        
        # PyTorch CTC Loss girdileri [T, B, V] formatında bekler
        outputs_ctc = outputs.permute(1, 0, 2).log_softmax(2)
        
        loss = ctc_loss(outputs_ctc, y.to(device), input_lens.to(device), target_lens.to(device))
        
        # Geri yayılım ve güncelleme
        loss.backward()
        optimizer.step()
        
        logger.info(f"Epoch {epoch}/5 — Kayıp (Loss): {loss.item():.4f}")
        
    logger.info("Eğitim başarıyla tamamlandı!")
    
    # 4. Modeli ONNX Formatına Aktarma
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/pi_model_float32.onnx"
    logger.info(f"Model ONNX formatına aktarılıyor: {onnx_path}")
    
    model.eval()
    # ONNX ihracı için örnek bir girdi [Batch=1, SeqLen=6, Channel=1, H=96, W=96]
    dummy_input = torch.randn(1, 6, 1, 96, 96, dtype=torch.float32).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    logger.info(f"🎉 Model başarıyla kaydedildi: {onnx_path}")

if __name__ == "__main__":
    main()
