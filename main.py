import sys
import os
import logging
from PyQt6.QtWidgets import QApplication
import yaml

from frontend.main_window import MainWindow


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log", encoding="utf-8"),
        ],
    )


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    setup_logging()
    cfg = load_config()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(
        model_path=cfg.get("model_path", "models/student_int8.onnx"),
        chunk_size=cfg.get("chunk_size", 6),
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
