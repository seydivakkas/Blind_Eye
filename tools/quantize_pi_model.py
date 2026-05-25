"""
tools/quantize_pi_model.py
Raspberry Pi 3 Model B+ için ultra-hafif Türkçe dudak okuma modelinin INT8 dinamik kuantizasyon betiği.
"""

import os
import sys
import time
import json
import logging
import numpy as np

# Logging ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def quantize_pi_model(fp32_path: str, int8_path: str):
    """
    ONNX formatındaki float32 modeli dinamik INT8 formatına kuantize eder.
    Bu işlem, Raspberry Pi 3 B+ gibi integer işlemlerinde daha hızlı olan gömülü işlemcilerde
    gecikmeyi (latency) azaltır ve bellek boyutunu 4 katına kadar küçültür.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime veya onnxruntime.quantization bulunamadı! Lütfen pip install onnxruntime yükleyin.")
        sys.exit(1)

    logger.info(f"Kuantizasyon işlemi başlatılıyor...")
    logger.info(f"Giriş Modeli (FP32): {fp32_path}")
    logger.info(f"Çıkış Modeli (INT8): {int8_path}")

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    duration = time.perf_counter() - t0
    logger.info(f"Kuantizasyon {duration:.2f} saniyede tamamlandı.")

    # Dosya boyutlarını karşılaştır
    size_fp32 = os.path.getsize(fp32_path) / (1024 ** 2)
    size_int8 = os.path.getsize(int8_path) / (1024 ** 2)
    shrink = (1 - size_int8 / size_fp32) * 100

    logger.info(f"✅ INT8 Kuantizasyon Tamamlandı!")
    logger.info(f"  FP32 Boyut: {size_fp32:.2f} MB")
    logger.info(f"  INT8 Boyut: {size_int8:.2f} MB")
    logger.info(f"  Boyut Azalması: %{shrink:.1f}")

def benchmark(onnx_path: str, dummy_input: np.ndarray, runs: int = 100) -> dict:
    """ONNX modelinin CPU üzerindeki ortalama ve p95 gecikmelerini hesaplar."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime bulunamadı!")
        return {}

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warm-up (Isınma turları)
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "min_ms": np.min(times) * 1000,
    }

def main():
    FP32_PATH = "models/pi_model_float32.onnx"
    INT8_PATH = "models/pi_model_int8.onnx"

    if not os.path.exists(FP32_PATH):
        logger.error(f"Float32 modeli bulunamadı: {FP32_PATH}. Lütfen önce train_pi_real.py betiğini çalıştırın.")
        sys.exit(1)

    # 1. Kuantize et
    quantize_pi_model(FP32_PATH, INT8_PATH)

    # 2. Benchmark yap
    # PiLipReadingModel girdi boyutu: [B, T, C, H, W] -> [1, 30, 1, 96, 96]
    dummy_input = np.random.randn(1, 30, 1, 96, 96).astype(np.float32)

    logger.info("\n📊 Performans Testleri Başlatılıyor...")
    results = {}
    for label, path in [("FP32", FP32_PATH), ("INT8", INT8_PATH)]:
        if os.path.exists(path):
            res = benchmark(path, dummy_input)
            if res:
                size_mb = os.path.getsize(path) / (1024 ** 2)
                results[label] = {
                    "latency_ms": res["mean_ms"],
                    "latency_p95_ms": res["p95_ms"],
                    "size_mb": size_mb,
                }
                logger.info(
                    f"  {label:15}: Ortalama = {res['mean_ms']:6.2f} ms | "
                    f"P95 = {res['p95_ms']:6.2f} ms | "
                    f"En Hızlı = {res['min_ms']:6.2f} ms | Boyut = {size_mb:.3f} MB"
                )

    # Sonuçları kaydet
    metrics_path = "results/pi_quant_metrics.json"
    os.makedirs("results", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n📊 Kuantizasyon performans metrikleri kaydedildi: {metrics_path}")
    logger.info("\n🎉 Raspberry Pi 3 B+ uyumlu kuantizasyon süreci başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
