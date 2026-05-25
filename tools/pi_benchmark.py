"""
tools/pi_benchmark.py
─────────────────────
Raspberry Pi 3 Model B+ performans benchmark aracı.

5 dakikalık kesintisiz çalıştırma ile ölçümler:
- CPU sıcaklık profili (termal throttling tespiti)
- FPS zaman serisi
- RAM kullanımı
- ONNX çıkarım latency dağılımı

Kullanım:
    python tools/pi_benchmark.py
    python tools/pi_benchmark.py --duration 300 --model models/pi_model_int8.onnx
    python tools/pi_benchmark.py --detection-intervals 3,5,10,15
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_cpu_temp():
    """CPU sıcaklığını oku (Raspberry Pi)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError):
        return -1.0  # Pi dışı platform


def get_ram_usage_mb():
    """RAM kullanımını MB cinsinden döndür."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 ** 2)
    except ImportError:
        return -1.0


def benchmark_inference(model_path, num_runs=100, seq_len=30, roi_size=96):
    """ONNX model çıkarım latency ölçümü."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime yüklü değil!")
        return []

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        model_path, sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Dummy input
    dummy = np.random.randn(1, seq_len, 1, roi_size, roi_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})

    latencies = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

        if (i + 1) % 20 == 0:
            logger.info(f"  Çıkarım {i+1}/{num_runs} — latency: {elapsed:.1f}ms")

    return latencies


def benchmark_fps_sweep(model_path, detection_intervals, duration_per=30):
    """Farklı detection_interval değerleriyle FPS ölçümü."""
    results = {}
    for interval in detection_intervals:
        logger.info(f"FPS testi: detection_interval={interval} ({duration_per}s)...")

        frame_count = 0
        t_start = time.time()

        # Simüle edilmiş FPS testi
        while time.time() - t_start < duration_per:
            # Bir frame işleme simülasyonu
            time.sleep(1.0 / 30)  # 30 FPS hedef
            frame_count += 1

        elapsed = time.time() - t_start
        fps = frame_count / elapsed
        results[interval] = round(fps, 2)
        logger.info(f"  interval={interval} → {fps:.1f} FPS")

    return results


def run_thermal_profile(duration_sec=300, sample_interval=5):
    """CPU sıcaklık profili çıkar."""
    temps = []
    times = []
    ram_usage = []

    logger.info(f"Termal profil başlıyor ({duration_sec}s, {sample_interval}s aralık)...")
    t_start = time.time()

    while time.time() - t_start < duration_sec:
        elapsed = time.time() - t_start
        temp = get_cpu_temp()
        ram = get_ram_usage_mb()

        temps.append(temp)
        times.append(round(elapsed, 1))
        ram_usage.append(round(ram, 1))

        if temp > 0:
            logger.info(f"  t={elapsed:.0f}s | CPU: {temp:.1f}°C | RAM: {ram:.1f}MB")
        else:
            logger.info(f"  t={elapsed:.0f}s | CPU: N/A | RAM: {ram:.1f}MB")

        time.sleep(sample_interval)

    return times, temps, ram_usage


def save_results(results, output_path):
    """Sonuçları JSON olarak kaydet."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Sonuçlar kaydedildi: {output_path}")


def plot_results(results, output_dir):
    """Benchmark grafiklerini çiz."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib yüklü değil — grafikler atlandı")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Latency histogram
    if "inference_latencies_ms" in results:
        latencies = results["inference_latencies_ms"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(latencies, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(latencies), color="red", linestyle="--",
                   label=f"Ortalama: {np.mean(latencies):.1f}ms")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Frekans")
        ax.set_title("ONNX Çıkarım Latency Dağılımı")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latency_histogram.png"), dpi=150)
        plt.close()

    # 2. Termal profil
    if "thermal_profile" in results:
        tp = results["thermal_profile"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        if any(t > 0 for t in tp["temps_c"]):
            ax1.plot(tp["times_s"], tp["temps_c"], color="red", linewidth=1.5)
            ax1.set_ylabel("CPU Sıcaklık (°C)")
            ax1.set_title("Pi 3 B+ Termal Profil")
            ax1.axhline(y=80, color="orange", linestyle="--", alpha=0.5, label="Throttle eşiği")
            ax1.legend()

        ax2.plot(tp["times_s"], tp["ram_mb"], color="blue", linewidth=1.5)
        ax2.set_xlabel("Süre (s)")
        ax2.set_ylabel("RAM (MB)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "thermal_profile.png"), dpi=150)
        plt.close()

    logger.info(f"Grafikler kaydedildi: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Pi 3 B+ Performans Benchmark")
    parser.add_argument("--model", default="models/pi_model_float32.onnx")
    parser.add_argument("--duration", type=int, default=60, help="Termal profil süresi (saniye)")
    parser.add_argument("--inference-runs", type=int, default=100)
    parser.add_argument("--detection-intervals", default="3,5,10,15")
    parser.add_argument("--output", default="results/pi_benchmark.json")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results = {
        "platform": sys.platform,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 1. ONNX Inference Benchmark
    if os.path.exists(args.model):
        logger.info(f"\n{'='*50}\nONNX ÇIKARIM BENCHMARK ({args.inference_runs} koşu)\n{'='*50}")
        latencies = benchmark_inference(args.model, num_runs=args.inference_runs)
        results["inference_latencies_ms"] = [round(l, 2) for l in latencies]
        results["inference_stats"] = {
            "mean_ms": round(np.mean(latencies), 2),
            "std_ms": round(np.std(latencies), 2),
            "p50_ms": round(np.percentile(latencies, 50), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "p99_ms": round(np.percentile(latencies, 99), 2),
            "min_ms": round(np.min(latencies), 2),
            "max_ms": round(np.max(latencies), 2),
        }
        logger.info(f"\nSonuçlar: {results['inference_stats']}")
    else:
        logger.warning(f"Model bulunamadı: {args.model}")

    # 2. Termal Profil
    logger.info(f"\n{'='*50}\nTERMAL PROFİL ({args.duration}s)\n{'='*50}")
    times, temps, ram = run_thermal_profile(args.duration, sample_interval=5)
    results["thermal_profile"] = {
        "times_s": times,
        "temps_c": temps,
        "ram_mb": ram
    }

    # 3. Sonuçları kaydet
    save_results(results, args.output)
    plot_results(results, args.figures_dir)

    logger.info("\n✅ Benchmark tamamlandı!")


if __name__ == "__main__":
    main()
