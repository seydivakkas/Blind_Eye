"""
pc/pc_main.py
═════════════
PC Pipeline Orchestrator — Tüm Modülleri Bağlar

Akış:
    StreamReceiver (MJPEG/HTTP)
        → Preprocessor (FaceMesh → ROI + landmarks)
            → VSREngine (DC-TCN ONNX GPU → logits)
            → FaceCueAnalyzer (landmarks → punctuation)
                → FusionDecoder (logits + cues → text)
                    → OledSender (MQTT → Pi)

Kullanım:
    python -m pc.pc_main --pi-url http://192.168.1.50:8080/video
    python -m pc.pc_main --model models/student_fp32.onnx --lm models/tr_3gram.arpa
    python -m pc.pc_main --webcam 0   # PC webcam ile test

Kısayollar:
    Ctrl+C → Güvenli kapatma
"""

import argparse
import logging
import os
import signal
import sys
import time
import threading
from collections import deque

import numpy as np

# Proje kökünü PATH'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc.stream_receiver import StreamReceiver
from pc.preprocess import Preprocessor, PreprocessResult
from pc.vsr_engine import VSREngine
from pc.face_cues import FaceCueAnalyzer
from pc.fusion_decoder import FusionDecoder
from pc.oled_sender import OledSender

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pc_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pc_main")

# ANSI renk kodları
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class PCPipeline:
    """PC tarafı VSR pipeline orchestrator.

    Parameters
    ----------
    pi_url : str
        Pi'nin MJPEG stream URL'si.
    model_path : str
        ONNX model dosya yolu.
    lm_path : str | None
        KenLM .arpa model yolu.
    broker_host : str
        MQTT broker IP (genelde PC'nin kendi IP'si).
    chunk_size : int
        VSR giriş frame sayısı (model'den otomatik algılanır).
    webcam : int | None
        PC webcam ile test (Pi yerine). None ise Pi stream kullanılır.
    """

    def __init__(
        self,
        pi_url: str = "http://192.168.1.50:8080/video",
        model_path: str = "models/student_fp32.onnx",
        lm_path: str = None,
        broker_host: str = "192.168.1.100",
        chunk_size: int = 30,
        webcam: int = None,
    ):
        self._running = False
        self._webcam_mode = webcam is not None
        self._webcam_id = webcam

        # ── Modülleri oluştur ──
        logger.info(f"{BOLD}═══ Blind Eye PC Pipeline ═══{RESET}")

        # 1. Stream Receiver
        if self._webcam_mode:
            self._receiver = None
            logger.info(f"📷 Webcam modu: ID={webcam}")
        else:
            self._receiver = StreamReceiver(pi_url=pi_url, buffer_size=90)

        # 2. Preprocessor
        self._preprocessor = Preprocessor(roi_size=(96, 96), margin=0.25)

        # 3. VSR Engine
        self._engine = VSREngine(model_path=model_path, chunk_size=chunk_size)

        # 4. Face Cue Analyzer
        self._face_cues = FaceCueAnalyzer(fps=15.0)

        # 5. Fusion Decoder
        self._decoder = FusionDecoder(
            lm_path=lm_path,
            beam_width=100,
            alpha=0.5,
            beta=1.0,
        )

        # 6. OLED Sender
        self._sender = OledSender(broker_host=broker_host)

        # Pipeline state
        self._roi_buffer: deque = deque(maxlen=self._engine.chunk_size)
        self._last_face_cue = None
        self._inference_thread = None
        self._frame_count = 0
        self._fps = 0.0
        self._fps_time = time.time()
        self._fps_counter = 0

    def start(self):
        """Pipeline'ı başlat."""
        if self._running:
            return

        os.makedirs("logs", exist_ok=True)
        self._running = True

        # Modülleri başlat
        if self._receiver:
            self._receiver.start()
        self._sender.start()

        # Ana pipeline thread
        self._inference_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True, name="pc-pipeline"
        )
        self._inference_thread.start()

        logger.info(f"{GREEN}✅ PC Pipeline başlatıldı{RESET}")
        logger.info(
            f"   Model    : {self._engine.model_path} "
            f"(T={self._engine.chunk_size}, V={self._engine.num_classes})"
        )
        logger.info(f"   Provider : {self._engine.active_provider}")
        logger.info(f"   Mock     : {self._engine.mock_mode}")

    def stop(self):
        """Pipeline'ı durdur."""
        logger.info("Pipeline durduruluyor...")
        self._running = False

        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)

        if self._receiver:
            self._receiver.stop()
        self._preprocessor.close()
        self._engine.close()
        self._sender.stop()

        logger.info(f"{GREEN}✅ Pipeline güvenli şekilde kapatıldı{RESET}")

    def _pipeline_loop(self):
        """Ana pipeline döngüsü."""
        import cv2  # Webcam modu için

        cap = None
        if self._webcam_mode:
            cap = cv2.VideoCapture(self._webcam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            if not cap.isOpened():
                logger.error("Webcam açılamadı!")
                return
            logger.info(f"Webcam açıldı: {self._webcam_id}")

        while self._running:
            try:
                # ── 1. Frame Al ──
                frame = None
                if self._webcam_mode and cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                elif self._receiver:
                    frame = self._receiver.get_latest()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # ── 2. Preprocess ──
                prep = self._preprocessor.process(frame)

                if not prep.face_detected or prep.roi is None:
                    time.sleep(0.01)
                    continue

                # ── 3. Face Cues (paralel) ──
                if prep.landmarks:
                    self._last_face_cue = self._face_cues.analyze(prep.landmarks)

                # ── 4. ROI Buffer → VSR Inference ──
                self._roi_buffer.append(prep.roi)

                if len(self._roi_buffer) < self._engine.chunk_size:
                    continue

                # Chunk oluştur: [T, 96, 96, 1]
                chunk = np.array(list(self._roi_buffer), dtype=np.float32)

                # VSR inference
                logits = self._engine.infer(chunk)

                if logits is None:
                    continue

                # ── 5. Fusion Decode ──
                result = self._decoder.decode(logits, self._last_face_cue)

                if result.text:
                    line1, line2 = self._decoder.get_subtitle_lines()

                    # MQTT → Pi
                    self._sender.send_subtitle(
                        line1=line1,
                        line2=line2,
                        confidence=result.confidence,
                        punctuation=result.punctuation,
                    )

                    # Console log
                    cue_info = f" [{result.punctuation}]" if result.face_cue_applied else ""
                    logger.info(
                        f"{CYAN}📝{RESET} \"{result.text}\" "
                        f"({result.confidence:.0%}) "
                        f"[{self._engine.last_latency_ms:.0f}ms VSR + "
                        f"{result.latency_ms:.0f}ms decode]{cue_info}"
                    )

                # FPS hesapla
                self._fps_counter += 1
                now = time.time()
                if now - self._fps_time >= 2.0:
                    self._fps = self._fps_counter / (now - self._fps_time)
                    self._fps_counter = 0
                    self._fps_time = now

                # Buffer'ı yarı yarıya kaydır (sliding window)
                half = self._engine.chunk_size // 2
                for _ in range(half):
                    if self._roi_buffer:
                        self._roi_buffer.popleft()

            except Exception as e:
                logger.error(f"Pipeline hatası: {e}", exc_info=True)
                time.sleep(0.1)

        # Temizlik
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Blind Eye — PC VSR Pipeline (ResNet18 + DC-TCN + CTC)"
    )
    parser.add_argument(
        "--pi-url", type=str, default="http://192.168.1.50:8080/video",
        help="Pi MJPEG stream URL'si",
    )
    parser.add_argument(
        "--model", type=str, default="models/student_fp32.onnx",
        help="ONNX model dosya yolu",
    )
    parser.add_argument(
        "--lm", type=str, default=None,
        help="KenLM .arpa model yolu (None = LM yok)",
    )
    parser.add_argument(
        "--broker", type=str, default="192.168.1.100",
        help="MQTT broker IP adresi (PC'nin IP'si)",
    )
    parser.add_argument(
        "--webcam", type=int, default=None,
        help="PC webcam ile test (Pi yerine). Örn: --webcam 0",
    )
    args = parser.parse_args()

    # KenLM otomatik algılama
    lm_path = args.lm
    if lm_path is None and os.path.exists("models/tr_3gram.arpa"):
        lm_path = "models/tr_3gram.arpa"
        logger.info(f"KenLM otomatik algılandı: {lm_path}")

    pipeline = PCPipeline(
        pi_url=args.pi_url,
        model_path=args.model,
        lm_path=lm_path,
        broker_host=args.broker,
        webcam=args.webcam,
    )

    # Ctrl+C handler
    def signal_handler(sig, frame):
        logger.info("\nKapatılıyor...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    pipeline.start()

    # Ana thread'i canlı tut
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
