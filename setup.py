#!/usr/bin/env python3
"""
setup.py — Blind Eye Otomatik Kurulum Scripti
──────────────────────────────────────────────
TÜBİTAK 2209-A | Türkçe Dudak Okuma Prototipi

Kullanım:
    python setup.py              # Masaüstü bağımlılıkları kur
    python setup.py --pi         # Pi 3 B+ minimal bağımlılıkları kur
    python setup.py --dev        # Geliştirme araçları dahil kur
    python setup.py --check      # Kurulum durumunu kontrol et
"""

import subprocess
import sys
import os
import platform
import argparse
import shutil

# Windows konsol encoding sorunu çözümü
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



# ═══════════════════════════════════════════════
#  Bağımlılık Listeleri
# ═══════════════════════════════════════════════

CORE_DEPS = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "onnxruntime>=1.16",
    "pyyaml>=6.0",
]

DESKTOP_DEPS = [
    "PyQt6>=6.5",
    "mediapipe>=0.10",
    "matplotlib>=3.7",
]

PI_DEPS = [
    # Pi 3 B+'da mediapipe yerine tflite kullanılır
    "tflite-runtime>=2.14",
]

PI_GLASSES_DEPS = [
    "luma.oled>=3.12",
    "Pillow>=10.0",
]

DEV_DEPS = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "flake8>=6.0",
    "mypy>=1.0",
    "black>=23.0",
]

TRAINING_DEPS = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "tqdm>=4.60",
    "tensorboard>=2.14",
]


def run_pip(packages: list[str], label: str):
    """pip ile paketleri yükler."""
    if not packages:
        return
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    subprocess.check_call(cmd)
    print(f"  ✅ {label} tamamlandı.\n")


def check_package(name: str) -> bool:
    """Paketin yüklü olup olmadığını kontrol eder."""
    try:
        __import__(name.split("[")[0].replace("-", "_"))
        return True
    except ImportError:
        return False


def check_installation():
    """Kurulum durumunu kontrol eder ve rapor verir."""
    print("\n" + "=" * 60)
    print("  Blind Eye — Kurulum Durum Raporu")
    print("=" * 60)

    print(f"\n  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")

    checks = {
        "numpy": "numpy",
        "OpenCV": "cv2",
        "ONNX Runtime": "onnxruntime",
        "PyYAML": "yaml",
        "PyQt6": "PyQt6",
        "MediaPipe": "mediapipe",
        "matplotlib": "matplotlib",
        "luma.oled": "luma.oled",
        "pytest": "pytest",
        "torch": "torch",
    }

    print("\n  Paket Durumu:")
    print("  " + "-" * 40)

    all_ok = True
    for display_name, import_name in checks.items():
        status = "✅" if check_package(import_name) else "❌"
        if status == "❌":
            all_ok = False
        print(f"  {status} {display_name:<20} ({import_name})")

    # Model dosyası kontrolü
    print("\n  Model Dosyaları:")
    print("  " + "-" * 40)
    models = [
        "models/student_int8.onnx",
        "models/pi_model_int8.onnx",
        "models/pi_model_float32.onnx",
    ]
    for model in models:
        if os.path.exists(model):
            size = os.path.getsize(model)
            print(f"  ✅ {model} ({size / 1024:.1f} KB)")
        else:
            print(f"  ⚠️  {model} (bulunamadı)")

    # Veri dosyaları
    print("\n  Veri Dosyaları:")
    print("  " + "-" * 40)
    data_files = [
        "data/processed/labels.json",
        "configs/default.yaml",
        "configs/training.yaml",
        "configs/tr_viseme_map.json",
        "configs/viseme_vocab.json",
    ]
    for df in data_files:
        status = "✅" if os.path.exists(df) else "⚠️ "
        print(f"  {status} {df}")

    print("\n" + "=" * 60)
    if all_ok:
        print("  🎉 Tüm temel bağımlılıklar yüklü!")
    else:
        print("  ⚠️  Bazı bağımlılıklar eksik. `python setup.py` çalıştırın.")
    print("=" * 60 + "\n")


def create_directories():
    """Gerekli dizinleri oluşturur."""
    dirs = [
        "logs",
        "results",
        "results/figures",
        "models/checkpoints",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Blind Eye Kurulum Scripti",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python setup.py              # Masaüstü kurulumu
  python setup.py --pi         # Pi 3 B+ kurulumu
  python setup.py --dev        # Geliştirme araçları dahil
  python setup.py --training   # Eğitim bağımlılıkları dahil
  python setup.py --check      # Kurulum durumu kontrol
        """
    )
    parser.add_argument("--pi", action="store_true", help="Pi 3 B+ minimal kurulum")
    parser.add_argument("--dev", action="store_true", help="Geliştirme araçlarını dahil et")
    parser.add_argument("--training", action="store_true", help="Eğitim bağımlılıklarını dahil et")
    parser.add_argument("--glasses", action="store_true", help="Pi gözlük bağımlılıkları dahil et")
    parser.add_argument("--check", action="store_true", help="Kurulum durumunu kontrol et")
    parser.add_argument("--all", action="store_true", help="Tüm bağımlılıkları kur")
    args = parser.parse_args()

    if args.check:
        check_installation()
        return

    print("\n" + "=" * 60)
    print("  🔧 Blind Eye — Otomatik Kurulum")
    print("  TÜBİTAK 2209-A | Türkçe Dudak Okuma Prototipi")
    print("=" * 60)

    # Dizinleri oluştur
    create_directories()

    # Temel bağımlılıklar (her zaman)
    run_pip(CORE_DEPS, "Temel Bağımlılıklar")

    if args.pi:
        run_pip(PI_DEPS, "Pi 3 B+ Bağımlılıkları")
    else:
        run_pip(DESKTOP_DEPS, "Masaüstü Bağımlılıkları")

    if args.glasses or args.all:
        run_pip(PI_GLASSES_DEPS, "Pi Gözlük Bağımlılıkları")

    if args.dev or args.all:
        run_pip(DEV_DEPS, "Geliştirme Araçları")

    if args.training or args.all:
        run_pip(TRAINING_DEPS, "Eğitim Bağımlılıkları")

    print("\n  ✅ Kurulum tamamlandı!")
    print("  Durumu kontrol etmek için: python setup.py --check")
    print("  Uygulamayı başlatmak için: python -m frontend.main")
    print()


if __name__ == "__main__":
    main()
