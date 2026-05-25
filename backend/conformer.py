import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModule(nn.Module):
    """Feed-Forward module in Conformer block (Half-Step)."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.act = nn.SiLU()  # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """Convolutional Module in Conformer block for local anomaly/context aggregation."""

    def __init__(self, d_model: int, kernel_size: int = 31, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        
        # Pointwise Conv 1 (GLU Projection)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # 1D Depthwise Conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        # GLU projection
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Local conv
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, T, C]
        return residual + x


class ConformerBlock(nn.Module):
    """Conformer block containing Multi-Head Self-Attention, Convolution, and FFNs."""

    def __init__(self, d_model: int, n_heads: int = 4, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.post_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. FFN 1 (Half Step)
        x = self.ffn1(x)
        
        # 2. Multi-Head Self-Attention
        residual = x
        x = self.attn_ln(x)
        x, _ = self.attn(x, x, x)
        x = self.attn_dropout(x)
        x = residual + x
        
        # 3. Convolution
        x = self.conv_module(x)
        
        # 4. FFN 2 (Half Step)
        x = self.ffn2(x)
        
        # 5. Post Normalization
        x = self.post_ln(x)
        return x


class ConformerEncoder(nn.Module):
    """Stack of Conformer Blocks for deep sequence extraction."""

    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
