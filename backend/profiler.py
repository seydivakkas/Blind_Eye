import time
import csv
import os
import logging
from collections import deque
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class Profiler:
    """Latency/FPS/CPU/RAM loglama ve CSV çıktısı."""

    def __init__(self, log_path: str = "logs/metrics.csv", window: int = 30):
        self.window = window
        self.latencies = deque(maxlen=window)
        self.log_path = log_path
        self._process = None
        self._init_csv()

        if PSUTIL_AVAILABLE:
            self._process = psutil.Process(os.getpid())

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "latency_ms", "fps", "cpu_percent", "memory_mb", "ece"]
                )

    def log(self, latency: float, ece: float = None) -> Dict:
        self.latencies.append(latency)
        fps = 1.0 / latency if latency > 0 else 0
        cpu = 0.0
        mem = 0.0

        if self._process:
            try:
                cpu = self._process.cpu_percent(interval=None)
                mem = self._process.memory_info().rss / 1024 ** 2
            except Exception:
                pass

        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    time.time(),
                    f"{latency * 1000:.2f}",
                    f"{fps:.1f}",
                    f"{cpu:.1f}",
                    f"{mem:.1f}",
                    f"{ece:.6f}" if ece is not None else "",
                ])
        except Exception:
            pass

        result = {
            "latency_ms": round(latency * 1000, 2),
            "fps": round(fps, 1),
            "cpu_percent": round(cpu, 1),
            "memory_mb": round(mem, 1),
        }
        if ece is not None:
            result["ece"] = round(ece, 6)
        return result

    def get_latest(self) -> Dict:
        if not self.latencies:
            return {"latency_ms": 0, "fps": 0, "cpu_percent": 0, "memory_mb": 0}
        lat = sum(self.latencies) / len(self.latencies)
        cpu = 0.0
        mem = 0.0
        if self._process:
            try:
                cpu = self._process.cpu_percent(interval=None)
                mem = self._process.memory_info().rss / 1024 ** 2
            except Exception:
                pass
        return {
            "latency_ms": round(lat * 1000, 2),
            "fps": round(1.0 / lat if lat > 0 else 0, 1),
            "cpu_percent": round(cpu, 1),
            "memory_mb": round(mem, 1),
        }
