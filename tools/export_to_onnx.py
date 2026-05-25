"""
tools/export_to_onnx.py
TÜBİTAK 2209-A — PyTorch → ONNX → INT8 Quantization Pipeline

Kullanım:
    python tools/export_to_onnx.py

Çıktı:
    models/student_fp32.onnx   — Standart hassasiyetli model
    models/student_int8.onnx   — INT8 quantize edilmiş model
"""

import os
import sys
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Bağımlılık kontrolü ──
try:
    import torch
    import torch.nn as nn
except ImportError:
    logger.error("PyTorch yüklü değil: pip install torch")
    sys.exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    logger.error("ONNX araçları yüklü değil: pip install onnx onnxruntime")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  1. MODEL TANIMI — CNN + LSTM + CTC Head
# ═══════════════════════════════════════════════════════════════

class LipReadModel(nn.Module):
    """
    Giriş : [B, T, H, W, C]  →  [Batch, TimeSteps, 96, 96, 1]
    Çıkış : [B, T, NumClasses]  (CTC Loss logits)

    Mimari: 3-katman CNN → Global Avg Pool → Bi-LSTM → Linear
    """

    def __init__(self, num_classes: int = 30, hidden_dim: int = 128):
        super().__init__()

        # Feature Extraction (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 48×48

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 24×24

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),            # 1×1
        )

        # Sequence Modeling (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classification Head
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        b, t, h, w, c = x.size()
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)                             # [B*T, 128, 1, 1]
        x = x.view(b, t, -1)                        # [B, T, 128]
        lstm_out, _ = self.lstm(x)                   # [B, T, Hidden*2]
        out = self.classifier(lstm_out)              # [B, T, NumClasses]
        return out


# ═══════════════════════════════════════════════════════════════
#  2. ONNX EXPORT
# ═══════════════════════════════════════════════════════════════

def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    """PyTorch modelini ONNX formatına çevirir."""
    logger.info(f"ONNX export başlıyor → {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    logger.info(f"✅ FP32 ONNX doğrulandı — {size_mb:.2f} MB")


# ═══════════════════════════════════════════════════════════════
#  3. INT8 QUANTIZATION
# ═══════════════════════════════════════════════════════════════

def quantize_int8(fp32_path: str, int8_path: str):
    """Dinamik INT8 quantization uygular."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info("INT8 dynamic quantization başlıyor...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )

    size_fp32 = os.path.getsize(fp32_path) / (1024 ** 2)
    size_int8 = os.path.getsize(int8_path) / (1024 ** 2)
    shrink = (1 - size_int8 / size_fp32) * 100

    logger.info(f"✅ INT8 tamamlandı — {size_int8:.2f} MB (FP32: {size_fp32:.2f} MB, %{shrink:.1f} küçülme)")


# ═══════════════════════════════════════════════════════════════
#  4. BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(onnx_path: str, dummy_np: np.ndarray, runs: int = 50) -> dict:
    """ONNX Runtime ile latency benchmark."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warm-up
    for _ in range(5):
        sess.run(None, {input_name: dummy_np})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_np})
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "p95_ms":  np.percentile(times, 95) * 1000,
        "min_ms":  np.min(times) * 1000,
    }


# ═══════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    BATCH      = 1
    TIME_STEPS = 6
    HEIGHT     = 96
    WIDTH      = 96
    CHANNELS   = 1

    # Vocab.json'dan num_classes oku (tutarlılık)
    import json
    _vocab_path = os.path.join(os.path.dirname(__file__), "..", "configs", "vocab.json")
    try:
        with open(_vocab_path, "r", encoding="utf-8") as f:
            NUM_CLS = json.load(f).get("num_classes", 31)
        logger.info(f"vocab.json'dan num_classes={NUM_CLS} okundu.")
    except FileNotFoundError:
        NUM_CLS = 31
        logger.warning(f"vocab.json bulunamadı, varsayılan num_classes={NUM_CLS}")

    FP32_PATH  = "models/student_fp32.onnx"
    INT8_PATH  = "models/student_int8.onnx"

    # 1. Model oluştur
    model = LipReadModel(num_classes=NUM_CLS)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parametreleri: {total_params:,}")

    # 2. Dummy input
    dummy_torch = torch.randn(BATCH, TIME_STEPS, HEIGHT, WIDTH, CHANNELS)
    dummy_np = dummy_torch.numpy()

    os.makedirs("models", exist_ok=True)

    # 3. Export
    try:
        export_to_onnx(model, dummy_torch, FP32_PATH)
    except Exception as e:
        logger.error(f"ONNX export hatası: {e}")
        sys.exit(1)

    # 4. Quantize
    try:
        quantize_int8(FP32_PATH, INT8_PATH)
    except Exception as e:
        logger.warning(f"Quantization atlandı: {e}")

    # 5. Benchmark
    logger.info("\n📊 Benchmark başlıyor...")
    for label, path in [("FP32", FP32_PATH), ("INT8", INT8_PATH)]:
        if os.path.exists(path):
            res = benchmark(path, dummy_np)
            logger.info(
                f"  {label}: mean={res['mean_ms']:.2f}ms  "
                f"P95={res['p95_ms']:.2f}ms  "
                f"min={res['min_ms']:.2f}ms"
            )

    logger.info("\n🎉 İşlem tamamlandı! models/ klasörünü kontrol edin.")
