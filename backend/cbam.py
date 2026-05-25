"""
backend/cbam.py
CBAM (Convolutional Block Attention Module) — Channel + Spatial Attention

Referans: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)

Student modele entegre edilerek attention distillation yapılabilir.
Teacher modelden CBAM attention haritaları alınarak student'a
feature-level bilgi aktarılır (AttnFD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """Kanal dikkat mekanizması — hangi feature map'lerin önemli olduğunu öğrenir.

    Global Average Pool + Global Max Pool → MLP → Sigmoid
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))

        attention = torch.sigmoid(avg_out + max_out)
        return attention.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Uzamsal dikkat mekanizması — görüntünün hangi bölgesinin önemli olduğunu öğrenir.

    Channel-wise AvgPool + MaxPool → Conv2d → Sigmoid
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return attention


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module.

    Channel Attention → Spatial Attention sırasıyla uygulanır.
    Attention haritaları distillation için erişilebilir.

    Kullanım:
        cbam = CBAM(channels=64)
        out, ch_att, sp_att = cbam(features, return_attention=True)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple:
        """
        Args:
            x: [B, C, H, W] feature map
            return_attention: True ise attention haritalarını da döndürür

        Returns:
            return_attention=False → refined feature map
            return_attention=True → (refined, channel_att, spatial_att)
        """
        # 1. Channel attention
        ch_att = self.channel_attention(x)
        x_ch = x * ch_att

        # 2. Spatial attention
        sp_att = self.spatial_attention(x_ch)
        refined = x_ch * sp_att

        if return_attention:
            return refined, ch_att, sp_att
        return refined


# ═══════════════════════════════════════════════════════════════
#  CBAM'lı LipRead Model
# ═══════════════════════════════════════════════════════════════

class LipReadModelWithCBAM(nn.Module):
    """CBAM entegreli CNN+LSTM dudak okuma modeli.

    Giriş : [B, T, H, W, C]  →  [Batch, TimeSteps, 96, 96, 1]
    Çıkış : [B, T, NumClasses]  (CTC logits)

    Her CNN bloğundan sonra CBAM uygulanır.
    """

    def __init__(self, num_classes: int = 31, hidden_dim: int = 128):
        super().__init__()

        # CNN Block 1 + CBAM
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(32, reduction=8)
        self.pool1 = nn.MaxPool2d(2, 2)  # 48×48

        # CNN Block 2 + CBAM
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(64, reduction=16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 24×24

        # CNN Block 3 + CBAM
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(128, reduction=16)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # CTC Head
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [B, T, H, W, C]
            return_attention: True ise tüm CBAM attention haritalarını döndürür

        Returns:
            logits: [B, T, NumClasses]
            attention_maps: (sadece return_attention=True) dict of attention maps
        """
        b, t, h, w, c = x.size()
        x = x.view(b * t, c, h, w)

        attention_maps = {}

        # Block 1
        x = self.conv1(x)
        if return_attention:
            x, ch1, sp1 = self.cbam1(x, return_attention=True)
            attention_maps["block1"] = {"channel": ch1, "spatial": sp1}
        else:
            x = self.cbam1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        if return_attention:
            x, ch2, sp2 = self.cbam2(x, return_attention=True)
            attention_maps["block2"] = {"channel": ch2, "spatial": sp2}
        else:
            x = self.cbam2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        if return_attention:
            x, ch3, sp3 = self.cbam3(x, return_attention=True)
            attention_maps["block3"] = {"channel": ch3, "spatial": sp3}
        else:
            x = self.cbam3(x)
        x = self.gap(x)

        # Reshape → LSTM
        x = x.view(b, t, -1)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)

        if return_attention:
            return logits, attention_maps
        return logits


# ═══════════════════════════════════════════════════════════════
#  DISTILLATION LOSS (AttnFD)
# ═══════════════════════════════════════════════════════════════

class AttnFDLoss(nn.Module):
    """Attention Feature Distillation Loss.

    Teacher CBAM attention haritalarını student'a transfer eder.
    L_total = L_CTC + λ₁ · L_feature + λ₂ · L_attention

    Referans: Blind_Eye_Erişimi.md Bölüm 2341
    """

    def __init__(
        self,
        lambda_feature: float = 1.0,
        lambda_attention: float = 0.5,
    ):
        super().__init__()
        self.lambda_feature = lambda_feature
        self.lambda_attention = lambda_attention

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_attention: dict,
        teacher_attention: dict,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> dict:
        """
        Returns:
            dict with 'total', 'ctc', 'feature', 'attention' losses
        """
        # 1. CTC Loss
        ctc_loss = F.ctc_loss(
            student_logits.log_softmax(dim=-1).permute(1, 0, 2),
            targets,
            torch.full((student_logits.size(0),), student_logits.size(1), dtype=torch.long),
            target_lengths,
            blank=0,
        )

        # 2. Feature MSE Loss (logits level)
        feature_loss = F.mse_loss(student_logits, teacher_logits.detach())

        # 3. Attention Cosine Loss
        attention_loss = torch.tensor(0.0, device=student_logits.device)
        num_blocks = 0

        for block_name in student_attention:
            if block_name in teacher_attention:
                # Spatial attention cosine similarity
                s_sp = student_attention[block_name]["spatial"].flatten(1)
                t_sp = teacher_attention[block_name]["spatial"].flatten(1).detach()
                att_cos = 1 - F.cosine_similarity(s_sp, t_sp, dim=1).mean()
                attention_loss = attention_loss + att_cos
                num_blocks += 1

        if num_blocks > 0:
            attention_loss = attention_loss / num_blocks

        # Total
        total = (
            ctc_loss
            + self.lambda_feature * feature_loss
            + self.lambda_attention * attention_loss
        )

        return {
            "total": total,
            "ctc": ctc_loss.item(),
            "feature": feature_loss.item(),
            "attention": attention_loss.item(),
        }


# ═══════════════════════════════════════════════════════════════
#  V2: ResNet-18 Frontend + CBAM + Conformer + CTC
# ═══════════════════════════════════════════════════════════════

class LipReadModelV2(nn.Module):
    """V2 Mimari: ResNet-18 + CBAM + Conformer + CTC.

    LiRA / AV-HuBERT mimarisinden esinlenilmiştir.
    Cross-lingual transfer learning için tasarlanmıştır.

    Giriş : [B, T, H=96, W=96, C=1]
    Çıkış : [B, T, num_classes]

    Özellikler:
        - ResNet-18 3D-conv visual frontend (ImageNet pretrained opsiyonel)
        - CBAM attention (frontend çıkışına)
        - Conformer encoder (self-attn + conv temporal modeling)
        - CTC head (Türkçe 31 sınıf)
        - Progressive unfreezing desteği
    """

    def __init__(
        self,
        num_classes: int = 31,
        d_model: int = 256,
        n_conf_layers: int = 4,
        n_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        pretrained_resnet: bool = False,
    ):
        super().__init__()
        from .visual_frontend import VisualFrontend
        from .conformer import ConformerEncoder

        # 1. Visual Frontend (ResNet-18 + 3D Conv)
        self.frontend = VisualFrontend(
            in_channels=1, pretrained_resnet=pretrained_resnet
        )
        frontend_dim = self.frontend.output_dim  # 512

        # 2. CBAM on frontend features (frame-wise)
        #    [B*T, 512] → reshape to [B*T, 512, 1, 1] for CBAM → back
        self.use_cbam = True
        self.cbam = CBAM(frontend_dim, reduction=16)

        # 3. Projection: 512 → d_model
        self.proj = nn.Sequential(
            nn.Linear(frontend_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 4. Conformer Encoder
        self.conformer = ConformerEncoder(
            d_model=d_model,
            n_layers=n_conf_layers,
            n_heads=n_heads,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        # 5. CTC Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

        # Config
        self.d_model = d_model
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C]
            return_attention: True ise CBAM attention haritalarını döndür

        Returns:
            logits: [B, T, num_classes]
        """
        B = x.size(0)

        # 1. Visual Frontend: [B, T, H, W, C] → [B, T, 512]
        features = self.frontend(x)
        _, T, feat_dim = features.size()

        # 2. CBAM (frame-wise)
        attention_maps = {}
        if self.use_cbam:
            # [B, T, 512] → [B*T, 512, 1, 1]
            feat_2d = features.view(B * T, feat_dim, 1, 1)
            if return_attention:
                feat_2d, ch_att, sp_att = self.cbam(feat_2d, return_attention=True)
                attention_maps["cbam"] = {"channel": ch_att, "spatial": sp_att}
            else:
                feat_2d = self.cbam(feat_2d)
            features = feat_2d.view(B, T, feat_dim)

        # 3. Projection: [B, T, 512] → [B, T, d_model]
        features = self.proj(features)

        # 4. Conformer: [B, T, d_model] → [B, T, d_model]
        features = self.conformer(features)

        # 5. CTC Head: [B, T, d_model] → [B, T, num_classes]
        logits = self.classifier(features)

        if return_attention:
            return logits, attention_maps
        return logits

    def freeze_frontend(self):
        """Frontend (ResNet-18) katmanlarını dondur — transfer learning için."""
        for param in self.frontend.parameters():
            param.requires_grad = False
        print("[V2] Frontend donduruldu (requires_grad=False)")

    def unfreeze_frontend(self, lr_scale: float = 0.1):
        """Frontend'i unfreeze et — fine-tuning için.

        Returns:
            Param grupları (optimizer'a verilmek üzere)
        """
        for param in self.frontend.parameters():
            param.requires_grad = True
        print(f"[V2] Frontend unfreeze edildi (lr_scale={lr_scale})")
        return [
            {"params": self.frontend.parameters(), "lr_scale": lr_scale},
            {"params": self.cbam.parameters()},
            {"params": self.proj.parameters()},
            {"params": self.conformer.parameters()},
            {"params": self.classifier.parameters()},
        ]

    def param_count(self) -> dict:
        """Modül bazlı parametre sayısını döndür."""
        counts = {}
        for name, module in [
            ("frontend", self.frontend),
            ("cbam", self.cbam),
            ("proj", self.proj),
            ("conformer", self.conformer),
            ("classifier", self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts
