"""
pi_run.py
──────────
Raspberry Pi 3 Model B+ (ARMv8 4-Core 1.4GHz) Hibrit KLT + FaceMesh
Gerçek Zamanlı Türkçe Dudak Okuma & Akademik Mimik Analiz Arayüzü.

Mimari:
    Thread 1 (Ana): Kamera → KLT Takip → ROI Çıkarma → HUD Render → cv2.imshow
    Thread 2 (Arka): Asenkron FaceMesh + Mimik Analizi (her N karede bir)
    Thread 3 (Arka): ONNX CTC Çıkarım + Türkçe Decoder

Akademik özellikler:
    - KLT Optik Akış + FaceMesh Hibrit Takip (Tracking-by-Detection)
    - Zamansal Kinematik Mimik Analizi (Hız/İvme türevleri, Mikro-ifade)
    - Duchenne (Samimi) Gülümseme Tespiti
    - EAR + PERCLOS Bilişsel Yük İndeksi
    - CPU Sıcaklık İzleme (Termal Throttling)
    - Foveated Rendering (Vektörel HUD, sıfır alpha blending)
    - KVKK-by-Design (RAM-Only, sıfır disk kaydı)

Kullanım:
    python pi_run.py                     # Sadece dudak okuma (minimal CPU)
    python pi_run.py --mimic             # Tam akademik analiz + HUD
    python pi_run.py --mimic --source 0  # PC webcam ile test

Akademik referanslar:
    - Lucas & Kanade (1981): Optical Flow
    - Soukupová & Čech (2016): EAR Blink Detection
    - Ekman & Friesen (1978): FACS
    - Gulati et al. (2020): Conformer
    - Google LiRA (Ma et al., 2021): Visual Speech Recognition
"""

import os
import sys
import time
import queue
import threading
import argparse
import logging
import cv2
import numpy as np

# Proje dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.decoder import TurkishCTCDecoder
except ImportError:
    class TurkishCTCDecoder:
        TURKISH_VOCAB = list("<blank>") + list("abcçdefgğhıijklmnoöprsştuüvyz ")
        def __init__(self, vocab=None, blank_idx=0):
            self.vocab = vocab or self.TURKISH_VOCAB
            self.blank = blank_idx
        def decode(self, logits):
            if logits is None: return "", 0.0
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            tokens = np.argmax(probs, axis=-1).squeeze()
            cleaned = []
            for t in tokens:
                if t != self.blank and (not cleaned or t != cleaned[-1]):
                    cleaned.append(int(t))
            text = "".join(self.vocab[i] if i < len(self.vocab) else " " for i in cleaned)
            conf = float(np.mean(np.max(probs, axis=-1)))
            return text.strip(), conf

try:
    from backend.expression_detector import ExpressionDetector
except ImportError:
    ExpressionDetector = None

try:
    from backend.optical_flow_tracker import OpticalFlowTracker
except ImportError:
    OpticalFlowTracker = None

try:
    from ui.hud_renderer import HUDRenderer
except ImportError:
    HUDRenderer = None

try:
    from backend.lm_decoder import LMDecoder
except ImportError:
    LMDecoder = None

try:
    from backend.viseme_decoder import VisemeDecoder
except ImportError:
    VisemeDecoder = None

try:
    from backend.gpio_alert import GPIOAlert
except ImportError:
    GPIOAlert = None

# Pi Gözlük modülleri
try:
    from pi.pi_camera import PiCamera
except ImportError:
    PiCamera = None

try:
    from pi.stream_client import StreamClient
except ImportError:
    StreamClient = None

try:
    from pi.subtitle_receiver import SubtitleReceiver
except ImportError:
    SubtitleReceiver = None

try:
    from pi.oled_display import OledDisplay
except ImportError:
    OledDisplay = None

# Logging ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ANSI Renk kodları
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

# MediaPipe Dudak İndisleri
LIP_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]


# ═══════════════════════════════════════════════════════════════
#  Thread 2: Asenkron FaceMesh + Mimik Analiz Worker
# ═══════════════════════════════════════════════════════════════

class FaceMeshWorker(threading.Thread):
    """Arka planda MediaPipe FaceMesh + 3 katmanlı mimik analiz çalıştırır.

    Ana döngünün FPS'ini düşürmemek için ayrı thread'de koşar.
    Sonuçlar callback ile paylaşılır.

    Optimizasyon:
        - queue.Queue(maxsize=1): Bayat kare birikimini önler
        - time.sleep(0.05): Thread sonunda CPU'ya nefes aldırır
    """

    def __init__(self, input_queue, callback_func):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.callback_func = callback_func
        self.running = False

    def run(self):
        try:
            import mediapipe as mp
            if ExpressionDetector is None:
                raise ImportError("ExpressionDetector bulunamadı!")
        except ImportError as e:
            logger.error(f"FaceMeshWorker başlatılamadı: {e}")
            return

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        expr_detector = ExpressionDetector()
        self.running = True
        logger.info(f"✅ {GREEN}FaceMesh + Kinematik + Bilişsel Yük Thread'i başlatıldı!{RESET}")

        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                landmarks = None
                expressions = None

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    expressions = expr_detector.detect(landmarks)

                self.callback_func(landmarks, expressions)
                self.input_queue.task_done()

                # CPU'ya nefes aldır (termal throttling koruması)
                time.sleep(0.05)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"FaceMeshWorker hatası: {e}")
                continue

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
#  ONNX Inference Pipeline (Thread 3)
# ═══════════════════════════════════════════════════════════════

class PiPipeline:
    """ONNX CTC çıkarım pipeline'ı — arka plan thread'inde koşar (Raspberry Pi 3 Model B+)."""

    def __init__(self, model_path: str, seq_len: int = 6, roi_size: int = 96):
        self.model_path = model_path
        self.seq_len = seq_len
        self.roi_size = roi_size

        logger.info(f"Model yükleniyor: {BLUE}{model_path}{RESET}")
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                model_path, sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
        except ImportError:
            logger.error("onnxruntime bulunamadı!")
            sys.exit(1)

        self.decoder = TurkishCTCDecoder()
        self.frame_queue = queue.Queue(maxsize=12)
        self.result_queue = queue.Queue()
        self.running = False

        self.last_text = ""
        self.last_conf = 0.0
        self.last_latency = 0.0

        # Test uyumluluğu için senkron tampon bellek
        self.frame_buffer = []

        logger.info(f"✅ {GREEN}ONNX Pipeline hazır (4-core paralel){RESET}")

    def preprocess_frame(self, frame: np.ndarray, roi_coords: tuple) -> np.ndarray:
        """ROI çıkar, griye çevir, normalize et."""
        x, y, w, h = roi_coords
        f_h, f_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(f_w, x + w), min(f_h, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.roi_size, self.roi_size), dtype=np.float32)

        if len(crop.shape) == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop

        crop_resized = cv2.resize(crop_gray, (self.roi_size, self.roi_size))
        return crop_resized.astype(np.float32) / 255.0

    def add_frame(self, frame_norm: np.ndarray):
        """Frame'i kuyruğa ve test tamponuna ekler."""
        if not self.frame_queue.full():
            self.frame_queue.put(frame_norm)
        self.frame_buffer.append(frame_norm)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)

    def clear_buffer(self):
        """Tamponları sıfırlar."""
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.frame_buffer.clear()
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        self.last_text = ""
        self.last_conf = 0.0

    def predict(self) -> tuple:
        """Senkron tahmin veya asenkron sonuç okuma."""
        if not self.running:
            if len(self.frame_buffer) < self.seq_len:
                return "", 0.0, 0.0
            input_data = np.stack(self.frame_buffer, axis=0)
            input_data = np.expand_dims(input_data, axis=1)
            input_data = np.expand_dims(input_data, axis=0)
            t0 = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: input_data})
            latency = (time.perf_counter() - t0) * 1000
            logits = outputs[0]
            decoded_text, confidence = self.decoder.decode(logits[0])
            return decoded_text, confidence, latency
        else:
            try:
                text, conf, lat = self.result_queue.get(timeout=1.0)
                return text, conf, lat
            except queue.Empty:
                return "", 0.0, 0.0

    def inference_worker(self):
        """Thread 3: Arka plan ONNX çıkarım işçisi."""
        buffer = []
        logger.info("Çıkarım thread'i başlatıldı.")

        while self.running:
            try:
                frame_norm = self.frame_queue.get(timeout=0.1)
                buffer.append(frame_norm)
                if len(buffer) > self.seq_len:
                    buffer.pop(0)
                if len(buffer) == self.seq_len:
                    input_data = np.stack(buffer, axis=0)
                    input_data = np.expand_dims(input_data, axis=1)
                    input_data = np.expand_dims(input_data, axis=0)
                    t0 = time.perf_counter()
                    outputs = self.session.run(None, {self.input_name: input_data})
                    latency = (time.perf_counter() - t0) * 1000
                    logits = outputs[0]
                    decoded_text, confidence = self.decoder.decode(logits[0])
                    if decoded_text:
                        self.result_queue.put((decoded_text, confidence, latency))
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Tahmin hatası: {e}")
                continue

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        logger.info("Pipeline thread'leri durduruldu.")

# Geriye dönük test uyumluluğu için alias
PiZeroPipeline = PiPipeline
PiZero2WPipeline = PiPipeline


# ═══════════════════════════════════════════════════════════════
#  Ana Hibrit Döngü (KLT + FaceMesh + HUD)
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pi 3 B+ — Hibrit KLT+FaceMesh Türkçe Dudak Okuma & Akademik Mimik Analiz"
    )
    parser.add_argument("--model", type=str, default="models/pi_model_int8.onnx",
                        help="ONNX model yolu")
    parser.add_argument("--source", type=str, default="0",
                        help="Kamera ID (0, 1) veya video dosyası")
    parser.add_argument("--roi-size", type=int, default=96,
                        help="ROI dudak kırpma boyutu")
    parser.add_argument("--seq-len", type=int, default=6,
                        help="Giriş dizi uzunluğu")
    parser.add_argument("--mimic", action="store_true",
                        help="Akademik mimik analizi + fütüristik HUD'ı etkinleştir")
    parser.add_argument("--detection-interval", type=int, default=5,
                        help="FaceMesh kaç karede bir çalışsın (KLT aralığı)")
    parser.add_argument("--width", type=int, default=640,
                        help="Kamera genişliği")
    parser.add_argument("--height", type=int, default=480,
                        help="Kamera yüksekliği")
    # Faz 3: LM Beam Search
    parser.add_argument("--lm", type=str, default=None,
                        help="KenLM .arpa model yolu (beam search için)")
    parser.add_argument("--beam-width", type=int, default=50,
                        help="Beam search genişliği (varsayılan: 50)")
    # Erişilebilirlik
    parser.add_argument("--gpio", action="store_true",
                        help="GPIO LED/buzzer/titreşim uyarılarını etkinleştir")
    # Gözlük Stream Modu
    parser.add_argument("--stream", action="store_true",
                        help="WiFi stream modu — kamerayı PC'ye gönder, altyazı al")
    parser.add_argument("--server-ip", type=str, default="192.168.1.100",
                        help="PC'nin IP adresi (stream modu)")
    parser.add_argument("--stream-port", type=int, default=8554,
                        help="Video stream portu")
    parser.add_argument("--subtitle-port", type=int, default=8555,
                        help="Altyazı alım portu")
    parser.add_argument("--oled", action="store_true",
                        help="SSD1306 OLED ekranda altyazı göster")
    # Kalibrasyon
    parser.add_argument("--calibrate", action="store_true",
                        help="30 saniyelik EAR kalibrasyon modu")
    parser.add_argument("--benchmark", action="store_true",
                        help="5 dakikalık performans benchmark modu")
    args = parser.parse_args()

    # ── 1. Pipeline Başlat ──
    pipeline = PiPipeline(args.model, seq_len=args.seq_len, roi_size=args.roi_size)

    # LM Decoder entegrasyonu (Faz 3)
    if args.lm and LMDecoder is not None:
        lm_decoder = LMDecoder(
            lm_path=args.lm,
            beam_width=args.beam_width
        )
        pipeline.decoder = lm_decoder
        logger.info(f"✅ {GREEN}LM Beam Search aktif (beam={args.beam_width}){RESET}")
    elif args.lm:
        logger.warning(f"{YELLOW}LMDecoder import edilemedi — greedy decoder kullanılacak{RESET}")

    pipeline.start()

    # ── 2. KLT Optik Akış Takipçi ──
    tracker = None
    if OpticalFlowTracker is not None:
        tracker = OpticalFlowTracker(
            detection_interval=args.detection_interval,
            max_drift=15.0,
            fb_threshold=2.0,
        )
        logger.info(f"✅ {GREEN}KLT Optik Akış Takipçi aktif (interval={args.detection_interval}){RESET}")
    else:
        logger.warning(f"{YELLOW}OpticalFlowTracker bulunamadı — salt FaceMesh modunda çalışılacak.{RESET}")

    # ── 3. HUD Renderer ──
    hud = None
    if HUDRenderer is not None:
        hud = HUDRenderer(panel_width=180)
        logger.info(f"✅ {GREEN}Fütüristik HUD Renderer aktif{RESET}")
    else:
        logger.warning(f"{YELLOW}HUDRenderer bulunamadı — basit HUD kullanılacak.{RESET}")

    # ── 3b. GPIO Alert ──
    gpio = None
    if args.gpio and GPIOAlert is not None:
        gpio = GPIOAlert()
        logger.info(f"✅ {GREEN}GPIO uyarı sistemi aktif{RESET}")
    elif args.gpio:
        gpio = GPIOAlert(mock=True) if GPIOAlert else None
        if gpio:
            logger.info(f"{YELLOW}GPIO mock modda çalışıyor{RESET}")

    # ── 3c. OLED Display (Gözlük Modu) ──
    oled = None
    if args.oled and OledDisplay is not None:
        oled = OledDisplay()
        oled.start()
        logger.info(f"✅ {GREEN}SSD1306 OLED ekran aktif{RESET}")
    elif args.oled:
        logger.warning(f"{YELLOW}OledDisplay import edilemedi{RESET}")

    # ── 3d. WiFi Stream Client (Gözlük Modu) ──
    stream_client = None
    subtitle_rx = None
    if args.stream:
        if StreamClient is not None:
            logger.info(f"🔗 WiFi Stream modu → {args.server_ip}:{args.stream_port}")
        else:
            logger.error(f"{RED}Stream modülleri bulunamadı!{RESET}")
            args.stream = False

    # ── 4. Asenkron FaceMesh Thread ──
    shared_landmarks = None
    shared_expressions = None
    lock = threading.Lock()

    def face_results_callback(landmarks, expressions):
        nonlocal shared_landmarks, shared_expressions
        with lock:
            shared_landmarks = landmarks
            shared_expressions = expressions

    # maxsize=1: Bayat kare birikimini önler
    face_queue = queue.Queue(maxsize=1)
    face_worker = None

    if args.mimic:
        face_worker = FaceMeshWorker(face_queue, face_results_callback)
        face_worker.start()

    # ── 5. Kamera Bağlantısı ──
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error(f"{RED}Kamera açılamadı! Bağlantıları kontrol edin.{RESET}")
        pipeline.stop()
        if face_worker:
            face_worker.stop()
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Kamera yayını: {frame_width}x{frame_height} @ 30 FPS")

    # ── 6. Sabit ROI Koordinatları (FaceMesh yokken fallback) ──
    visible_w = frame_width - 180 if args.mimic else frame_width
    roi_w, roi_h = 120, 90
    roi_x = (visible_w - roi_w) // 2
    roi_y = (frame_height - roi_h) // 2
    static_roi_coords = (roi_x, roi_y, roi_w, roi_h)

    logger.info("Kısayollar: 'q' → Çıkış, 'c' → Tampon Sıfırla")

    # FPS sayacı
    fps_last_time = time.time()
    fps = 0
    frame_count = 0

    # Takip durumu
    tracking_mode = "detection"
    tracking_quality = 0.0
    current_roi_bbox = None  # (x1, y1, x2, y2) formatı

    # ═══════════════════════════════════════════════════════
    #  ANA HİBRİT DÖNGÜ
    # ═══════════════════════════════════════════════════════

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video akışı sonlandı.")
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── A. Paylaşılan FaceMesh verilerini oku (non-blocking) ──
            local_landmarks = None
            local_expr = None
            with lock:
                if shared_landmarks is not None:
                    local_landmarks = list(shared_landmarks)
                if shared_expressions is not None:
                    local_expr = dict(shared_expressions)

            # ── B. FaceMesh kuyruğuna gönder (maxsize=1 → bayat kare yok) ──
            if face_worker and args.mimic:
                # Tracker varsa sadece detection gerektiğinde gönder
                should_send = True
                if tracker is not None:
                    should_send = tracker.needs_detection
                # Tracker yoksa her 2 karede bir gönder
                elif frame_count % 2 != 0:
                    should_send = False

                if should_send:
                    # Eski kareyi at, yenisini koy (maxsize=1)
                    try:
                        face_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        face_queue.put_nowait(frame)
                    except queue.Full:
                        pass

            # ── C. KLT Hibrit Takip ──
            lip_landmarks_px = []
            all_landmarks_px = []

            if tracker is not None and args.mimic:
                if local_landmarks is not None and tracker.needs_detection:
                    # Yeni FaceMesh verisi geldi → anchor güncelle
                    tracker.update(gray, facemesh_landmarks=local_landmarks,
                                   frame_w=w, frame_h=h)
                    tracking_mode = "detection"
                else:
                    # KLT ile takip et
                    tracker.update(gray, frame_w=w, frame_h=h)
                    tracking_mode = "tracking"

                tracking_quality = tracker.tracking_quality

                # Dudak noktalarını al
                lip_pts = tracker.get_lip_points()
                if lip_pts is not None:
                    lip_landmarks_px = [(int(p[0]), int(p[1])) for p in lip_pts]

                    # Dinamik ROI hesapla
                    xs = lip_pts[:, 0]
                    ys = lip_pts[:, 1]
                    margin_w = int((np.max(xs) - np.min(xs)) * 0.20)
                    margin_h = int((np.max(ys) - np.min(ys)) * 0.20)
                    x1 = max(0, int(np.min(xs)) - margin_w)
                    y1 = max(0, int(np.min(ys)) - margin_h)
                    x2 = min(w, int(np.max(xs)) + margin_w)
                    y2 = min(h, int(np.max(ys)) + margin_h)
                    current_roi_bbox = (x1, y1, x2, y2)

                # Tüm yüz noktaları (mesh overlay için)
                if local_landmarks:
                    all_landmarks_px = [
                        (int(lm.x * w), int(lm.y * h)) for lm in local_landmarks
                    ]

            elif local_landmarks:
                # Tracker yok ama FaceMesh var → direkt kullan
                lip_landmarks_px = [
                    (int(local_landmarks[i].x * w), int(local_landmarks[i].y * h))
                    for i in LIP_OUTER
                ]
                xs = [p[0] for p in lip_landmarks_px]
                ys = [p[1] for p in lip_landmarks_px]
                if xs and ys:
                    margin_w = int((max(xs) - min(xs)) * 0.20)
                    margin_h = int((max(ys) - min(ys)) * 0.20)
                    x1 = max(0, min(xs) - margin_w)
                    y1 = max(0, min(ys) - margin_h)
                    x2 = min(w, max(xs) + margin_w)
                    y2 = min(h, max(ys) + margin_h)
                    current_roi_bbox = (x1, y1, x2, y2)

                all_landmarks_px = [
                    (int(lm.x * w), int(lm.y * h)) for lm in local_landmarks
                ]
                tracking_mode = "detection"
                tracking_quality = 1.0

            # ── D. ROI Çıkar + Çıkarım Kuyruğuna Gönder ──
            if current_roi_bbox is not None:
                bx1, by1, bx2, by2 = current_roi_bbox
                roi_coords = (bx1, by1, bx2 - bx1, by2 - by1)
            else:
                roi_coords = static_roi_coords

            roi_norm = pipeline.preprocess_frame(frame, roi_coords)
            if not pipeline.frame_queue.full():
                pipeline.frame_queue.put(roi_norm)

            # ── E. Çıkarım Sonuçlarını Oku (non-blocking) ──
            try:
                while not pipeline.result_queue.empty():
                    text, conf, lat = pipeline.result_queue.get_nowait()
                    pipeline.last_text = text
                    pipeline.last_conf = conf
                    pipeline.last_latency = lat
                    logger.info(
                        f"Tahmin: {BOLD}{GREEN}{text}{RESET} "
                        f"(Güven: {conf:.2f}, Çıkarım: {lat:.1f}ms)"
                    )
                    # OLED'de göster (Gözlük Modu)
                    if oled and text:
                        oled.show_subtitle(text, conf)
                    # GPIO tahmin onayı
                    if gpio and text:
                        gpio.confirm_prediction(text)
            except queue.Empty:
                pass

            # FPS hesabı
            frame_count += 1
            curr_time = time.time()
            if curr_time - fps_last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_last_time = curr_time

            # ── F. HUD Render ──
            display_frame = frame.copy()

            if hud is not None and args.mimic:
                hud.render(
                    frame=display_frame,
                    roi_bbox=current_roi_bbox,
                    lip_landmarks=lip_landmarks_px,
                    all_landmarks_px=all_landmarks_px if len(all_landmarks_px) > 0 else None,
                    expressions=local_expr,
                    tracking_mode=tracking_mode,
                    tracking_quality=tracking_quality,
                    fps=fps,
                    inference_latency=pipeline.last_latency,
                    prediction_text=pipeline.last_text,
                    prediction_conf=pipeline.last_conf,
                    mimic_mode=True,
                )
            else:
                # Minimal HUD (--mimic olmadan)
                if current_roi_bbox is not None:
                    bx1, by1, bx2, by2 = current_roi_bbox
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                else:
                    rx, ry, rw, rh = static_roi_coords
                    cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                    cv2.putText(display_frame, "DUDAKLARI HIZALAYIN", (rx - 15, ry - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                # Minimal sistem bilgisi
                cv2.putText(display_frame, f"FPS: {fps}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, "KVKK: RAM-Only", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

                # Alt bar tahmin
                cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)
                pred_msg = (
                    f"TURKCE: '{pipeline.last_text}' ({pipeline.last_conf * 100:.1f}%)"
                    if pipeline.last_text
                    else "Konusma bekleniyor..."
                )
                pred_color = (0, 255, 0) if pipeline.last_text else (150, 150, 150)
                cv2.putText(display_frame, pred_msg, (15, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1, cv2.LINE_AA)

            # ── G. Görüntüle ──
            cv2.imshow("Pi 3 B+ — Dudak Okuma & Mimik Analiz", display_frame)

            # ── H. Tuş Kontrolleri ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                pipeline.clear_buffer()
                if tracker:
                    tracker.reset()
                logger.info("Tamponlar sıfırlandı.")

    except KeyboardInterrupt:
        logger.info("Uygulama kullanıcı tarafından kapatıldı.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.stop()
        if face_worker:
            face_worker.stop()
        if oled:
            oled.stop()
        if stream_client:
            stream_client.stop()
        if subtitle_rx:
            subtitle_rx.stop()
        if gpio:
            gpio.cleanup()
        logger.info(f"✅ {GREEN}Sistem güvenli şekilde kapatıldı.{RESET}")


if __name__ == "__main__":
    main()
