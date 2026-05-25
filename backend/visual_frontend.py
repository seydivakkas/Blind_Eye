"""
backend/visual_frontend.py
──────────────────────────
ResNet-18 tabanlı Visual Frontend — LiRA/AV-HuBERT mimarisinden esinlenildi.

Yapı:
    1. 3D Conv katmanı  → spatio-temporal özellik çıkarma (1 → 64 kanal)
    2. ResNet-18 (2D)   → frame-wise zengin feature vektörleri (512-dim)
    3. Adaptive Pool    → sabit boyut çıktı

Giriş:  [B, T, H=96, W=96, C=1]
Çıkış:  [B, T, 512]

Referans:
    - Stafylakis & Tzimiropoulos, "Combining Residual Networks with LSTMs for Lipreading" (2017)
    - Ma et al., "LiRA: Learning Visual Speech Representations from Audio" (Interspeech 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """ResNet Basic Block (2 × 3×3 conv + skip connection)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class VisualFrontend(nn.Module):
    """ResNet-18 tabanlı visual frontend.

    LiRA stilinde:
        - İlk katman: 3D conv ile temporal bilgiyi yakalama
        - Sonrası: 2D ResNet-18 ile spatial feature çıkarma

    Args:
        in_channels: Giriş kanal sayısı (1=grayscale, 3=RGB)
        pretrained_resnet: True ise torchvision ResNet-18 ağırlıklarını yükle
                          (sadece in_channels=3 için geçerli)
    """

    def __init__(self, in_channels: int = 1, pretrained_resnet: bool = False):
        super().__init__()
        self.in_channels = in_channels

        # ── 3D Conv (Spatio-Temporal) ─────────────────────────────────────
        # LiRA: Conv3d(1, 64, kernel=(5,7,7), stride=(1,2,2))
        # 5 frame temporal kernel → dudak hareketi yakalanır
        self.frontend_3d = nn.Sequential(
            nn.Conv3d(
                in_channels, 64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # ── ResNet-18 (2D, frame-wise) ───────────────────────────────────
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Ağırlık başlatma
        self._init_weights()

        # Opsiyonel: torchvision ResNet-18 ağırlıklarını yükle
        if pretrained_resnet:
            self._load_pretrained_resnet18()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_resnet18(self):
        """torchvision ResNet-18 ağırlıklarını ResNet katmanlarına yükle."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)

            # Layer1-4 ağırlıklarını kopyala
            mapping = {
                "layer1": pretrained.layer1,
                "layer2": pretrained.layer2,
                "layer3": pretrained.layer3,
                "layer4": pretrained.layer4,
            }
            for name, pretrained_layer in mapping.items():
                target = getattr(self, name)
                src_state = pretrained_layer.state_dict()
                tgt_state = target.state_dict()

                # Boyut uyumlu olanları kopyala
                compatible = {}
                for k, v in src_state.items():
                    if k in tgt_state and v.shape == tgt_state[k].shape:
                        compatible[k] = v

                if compatible:
                    target.load_state_dict(compatible, strict=False)

            print(f"[VisualFrontend] ResNet-18 ImageNet agirliklari yuklendi")
        except Exception as e:
            print(f"[VisualFrontend] Pretrained yuklenemedi (devam ediliyor): {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] — video frame chunk'ı

        Returns:
            features: [B, T, 512] — frame bazlı feature vektörleri
        """
        B, T, H, W, C = x.size()

        # [B, T, H, W, C] → [B, C, T, H, W] (Conv3D formatı)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # 3D Conv: [B, C, T, H, W] → [B, 64, T, H', W']
        x = self.frontend_3d(x)

        # [B, 64, T, H', W'] → [B*T, 64, H', W'] (frame-wise 2D ResNet)
        _, C_out, T_out, H_out, W_out = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T_out, C_out, H_out, W_out)

        # ResNet katmanları
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pool → [B*T, 512, 1, 1] → [B*T, 512]
        x = self.avgpool(x)
        x = x.view(B * T_out, -1)

        # [B*T, 512] → [B, T, 512]
        x = x.view(B, T_out, -1)

        return x

    @property
    def output_dim(self) -> int:
        return 512
