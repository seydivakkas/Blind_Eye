"""
tests/test_profiler.py — Profiler birim testleri.
"""
import pytest
import csv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.profiler import Profiler


class TestCSVInit:
    def test_csv_created(self, tmp_path):
        """CSV dosyası oluşturulmalı."""
        log_file = tmp_path / "metrics.csv"
        Profiler(log_path=str(log_file))
        assert log_file.exists()

    def test_csv_headers(self, tmp_path):
        """CSV başlıkları doğru olmalı."""
        log_file = tmp_path / "metrics.csv"
        Profiler(log_path=str(log_file))
        with open(log_file) as f:
            header = next(csv.reader(f))
            assert "timestamp" in header
            assert "latency_ms" in header
            assert "fps" in header
            assert "cpu_percent" in header
            assert "memory_mb" in header


class TestLogging:
    def test_log_returns_dict(self, tmp_path):
        """log() dict döndürmeli."""
        p = Profiler(log_path=str(tmp_path / "m.csv"))
        result = p.log(0.05)
        assert isinstance(result, dict)
        assert "latency_ms" in result
        assert "fps" in result

    def test_log_appends_row(self, tmp_path):
        """Her log() çağrısı CSV'ye satır eklemeli."""
        log_file = tmp_path / "m.csv"
        p = Profiler(log_path=str(log_file))
        p.log(0.05)
        p.log(0.10)

        with open(log_file) as f:
            rows = list(csv.reader(f))
            assert len(rows) == 3  # header + 2 data rows

    def test_latency_average(self, tmp_path):
        """get_latest() doğru ortalama hesaplamalı."""
        p = Profiler(log_path=str(tmp_path / "m.csv"), window=3)
        p.log(0.050)   # 50ms
        p.log(0.100)   # 100ms
        p.log(0.150)   # 150ms
        latest = p.get_latest()
        # Ortalama: (50+100+150)/3 = 100ms
        assert abs(latest["latency_ms"] - 100.0) < 1.0


class TestWindowBehavior:
    def test_window_overflow(self, tmp_path):
        """Window dolduğunda eski veriler atılmalı."""
        p = Profiler(log_path=str(tmp_path / "m.csv"), window=2)
        p.log(0.01)
        p.log(0.02)
        p.log(0.03)  # ilk veri (0.01) atılmalı

        latest = p.get_latest()
        # Ortalama: (20+30)/2 = 25ms
        assert abs(latest["latency_ms"] - 25.0) < 1.0

    def test_empty_get_latest(self, tmp_path):
        """Veri yokken get_latest() sıfır döndürmeli."""
        p = Profiler(log_path=str(tmp_path / "m.csv"))
        latest = p.get_latest()
        assert latest["latency_ms"] == 0
        assert latest["fps"] == 0
