import threading
import queue
import time
import logging
import cv2
import numpy as np
from collections import deque
from PyQt6.QtCore import QObject, pyqtSignal

from .camera_manager import CameraManager
from .roi_processor import ROIProcessor
from .inference_engine import InferenceEngine
from .decoder import TurkishCTCDecoder
from .profiler import Profiler
from .stream_server import StreamServer

logger = logging.getLogger(__name__)


class MockCamera:
    """Kamera yoksa rastgele frame üreten yedek."""

    def start(self) -> bool:
        return True

    def read(self):
        time.sleep(0.033)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return True, frame

    def stop(self):
        pass


class PipelineController(QObject):
    """
    Backend orkestratörü.
    2 daemon thread + queue backpressure + pyqtSignal.
    
    Çalışma Modları:
    - "local": PC kamerası ile doğrudan çalışma (geliştirme/test)
    - "stream": WiFi üzerinden Pi gözlükten frame alma
    
    Kamera/Model yoksa otomatik mock moda geçer.
    """

    frame_ready = pyqtSignal(object)
    subtitle_ready = pyqtSignal(str, float)
    metrics_ready = pyqtSignal(dict)
    expression_ready = pyqtSignal(dict)
    tracking_quality = pyqtSignal(float, str)  # quality, mode
    pipeline_stopped = pyqtSignal()
    status_changed = pyqtSignal(str)
    stream_connected = pyqtSignal(bool)  # WiFi bağlantı durumu

    def __init__(self, model_path: str = "models/student_int8.onnx",
                 chunk_size: int = 6, fps_limit: int = 30,
                 mode: str = "local",
                 stream_port: int = 8554, subtitle_port: int = 8555):
        super().__init__()
        self.chunk_size = chunk_size
        self.mode = mode  # "local" | "stream"

        self.camera = CameraManager(fps=fps_limit)
        self.roi_proc = ROIProcessor()
        self.infer = InferenceEngine(model_path, chunk_size=chunk_size)
        self.decoder = TurkishCTCDecoder()
        self.profiler = Profiler()

        self.frame_q = queue.Queue(maxsize=2)
        self.roi_q = queue.Queue(maxsize=10)

        self.running = False
        self.threads = []
        self._use_mock_camera = False

        # WiFi Stream Server (sadece stream modunda)
        self._stream_server = None
        self._stream_port = stream_port
        self._subtitle_port = subtitle_port
        if mode == "stream":
            self._stream_server = StreamServer(
                port=stream_port, subtitle_port=subtitle_port
            )
            self._stream_server.on_frame = self._on_stream_frame

    def start(self) -> bool:
        if self.running:
            return True

        if self.mode == "stream":
            # WiFi stream modu — Pi'den frame beklenir
            logger.info("WiFi Stream modu — Pi bağlantısı bekleniyor...")
            self.status_changed.emit("WiFi Bekleniyor")
            if self._stream_server:
                self._stream_server.start()
        else:
            # Local kamera modu
            if not self.camera.start():
                logger.warning("Kamera bulunamadı — MockCamera aktif.")
                self.camera = MockCamera()
                self.camera.start()
                self._use_mock_camera = True
                self.status_changed.emit("Mock Kamera Aktif")
            else:
                self.status_changed.emit("Kamera Bağlı")

        self.running = True

        if self.mode == "local":
            t_cap = threading.Thread(
                target=self._capture_roi_loop, name="CapThread", daemon=True
            )
            self.threads.append(t_cap)
            t_cap.start()

        t_inf = threading.Thread(
            target=self._inference_loop, name="InfThread", daemon=True
        )
        self.threads.append(t_inf)
        t_inf.start()

        logger.info(f"Pipeline başlatıldı (mod={self.mode}).")
        self.status_changed.emit(f"Pipeline Çalışıyor ({self.mode})")
        return True

    def stop(self):
        if not self.running:
            return
        logger.info("Pipeline durduruluyor...")
        self.running = False

        if self.mode == "stream" and self._stream_server:
            self._stream_server.stop()
        else:
            self.camera.stop()

        for t in self.threads:
            t.join(timeout=2.0)
        self.threads.clear()

        for q in (self.frame_q, self.roi_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self.pipeline_stopped.emit()
        self.status_changed.emit("Pipeline Durduruldu")
        logger.info("Pipeline temizlendi ve durdu.")

    def _on_stream_frame(self, frame: np.ndarray):
        """
        WiFi stream'den gelen frame — capture_roi_loop'un stream versiyonu.
        StreamServer callback'i olarak çağrılır.
        """
        if not self.running:
            return

        # Bağlantı durumu sinyali
        if self._stream_server and self._stream_server.is_connected:
            self.stream_connected.emit(True)
            self.status_changed.emit(
                f"Pi Bağlı ({self._stream_server.actual_fps:.0f} fps)"
            )

        # ROI çıkar (pipeline.py'nin mevcut akışı ile aynı)
        roi = self.roi_proc.process(frame)

        # HUD overlay çiz (yerel gösterim için)
        annotated_frame = frame.copy()
        self._draw_hud(annotated_frame)

        try:
            self.frame_q.put_nowait(frame.copy())
            self.frame_ready.emit(annotated_frame)
            if roi is not None:
                self.expression_ready.emit(self.roi_proc.last_expressions)
        except queue.Full:
            pass

        self.tracking_quality.emit(
            self.roi_proc.last_tracking_quality,
            self.roi_proc.last_tracking_mode,
        )

        if roi is not None:
            try:
                self.roi_q.put_nowait(roi)
            except queue.Full:
                try:
                    self.roi_q.get_nowait()
                    self.roi_q.put_nowait(roi)
                except queue.Empty:
                    pass
        elif self.infer.mock_mode:
            mock_roi = np.random.rand(96, 96, 1).astype(np.float32)
            try:
                self.roi_q.put_nowait(mock_roi)
            except queue.Full:
                try:
                    self.roi_q.get_nowait()
                    self.roi_q.put_nowait(mock_roi)
                except queue.Empty:
                    pass

    # ── Duygu → BGR renk eşlemesi (Foveated HUD) ──
    _EMOTION_BGR = {
        "Gülümseme": (204, 255, 0),    # Neon cyan-yeşil
        "Kaş Çatma": (85, 34, 255),     # Neon kırmızı
        "Şaşırma": (0, 170, 255),       # Neon turuncu
        "Nötr": (255, 136, 68),          # Soğuk mavi
    }

    def _capture_roi_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Önce ROI hesapla ki bounding box ve landmarks güncellensin
            roi = self.roi_proc.process(frame)

            # ── Foveated Rendering: Vektörel HUD Overlay ──
            annotated_frame = frame.copy()
            if roi is not None and self.roi_proc.last_bbox is not None:
                x1, y1, x2, y2 = self.roi_proc.last_bbox

                # Duygu-reaktif renk belirleme
                expr = self.roi_proc.last_expressions
                dominant = expr.get("dominant", "Nötr") if expr else "Nötr"
                hud_color = self._EMOTION_BGR.get(dominant, (255, 136, 68))

                # Foveated köşe işaretleri (corner markers) — tam dikdörtgen yerine
                corner_len = max(12, (x2 - x1) // 5)
                thickness = 2
                # Sol üst
                cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), hud_color, thickness)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), hud_color, thickness)
                # Sağ üst
                cv2.line(annotated_frame, (x2, y1), (x2 - corner_len, y1), hud_color, thickness)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_len), hud_color, thickness)
                # Sol alt
                cv2.line(annotated_frame, (x1, y2), (x1 + corner_len, y2), hud_color, thickness)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_len), hud_color, thickness)
                # Sağ alt
                cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), hud_color, thickness)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), hud_color, thickness)

                # Dudak mesh — ince vektörel polyline (cv2.addWeighted yerine)
                lip_pts = self.roi_proc.last_landmarks
                if lip_pts and len(lip_pts) > 2:
                    pts = np.array(lip_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=True, color=hud_color, thickness=1, lineType=cv2.LINE_AA)

                # HUD etiket — duygu + güven
                if expr:
                    conf = expr.get("confidence", 0.0)
                    labels_map = {"Gülümseme": "MUTLU", "Kaş Çatma": "KIZGIN", "Şaşırma": "SASIRMIS", "Nötr": "NOTR"}
                    lbl_text = f"{labels_map.get(dominant, 'NOTR')} %{int(conf * 100)}"

                    # Mikro-ifade flash (magenta overlay)
                    kinematic = expr.get("kinematic", {})
                    micro = kinematic.get("micro_expression")
                    if micro:
                        lbl_text += f" [{micro}]"
                        hud_color = (255, 0, 255)  # Magenta flash

                    # Duchenne göstergesi
                    is_duchenne = kinematic.get("is_duchenne", False)
                    if is_duchenne:
                        lbl_text += " [D]"

                    label_y = max(y1 - 10, 25)
                    (w_text, h_text), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Arka plan kutusu (koyu lacivert) + HUD çerçeve
                    cv2.rectangle(annotated_frame, (x1, label_y - h_text - 6), (x1 + w_text + 12, label_y + 4), (26, 14, 10), -1)
                    cv2.rectangle(annotated_frame, (x1, label_y - h_text - 6), (x1 + w_text + 12, label_y + 4), hud_color, 1)
                    cv2.putText(annotated_frame, lbl_text, (x1 + 6, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)

                    # Bilişsel yük HUD (sağ alt köşe)
                    cognitive = expr.get("cognitive", {})
                    fatigue = cognitive.get("fatigue_level", "")
                    ear = cognitive.get("ear", 0.0)
                    if fatigue:
                        cog_text = f"EAR:{ear:.2f} [{fatigue}]"
                        cog_y = y2 + 18
                        cog_color = (0, 197, 34) if fatigue == "Optimal" else (0, 179, 234) if fatigue == "Normal" else (68, 68, 239)  # BGR
                        cv2.putText(annotated_frame, cog_text, (x1, cog_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cog_color, 1, cv2.LINE_AA)

            # Takip kalitesi sinyali
            self.tracking_quality.emit(
                self.roi_proc.last_tracking_quality,
                self.roi_proc.last_tracking_mode,
            )

            try:
                self.frame_q.put_nowait(frame.copy())
                self.frame_ready.emit(annotated_frame)
                if roi is not None:
                    self.expression_ready.emit(self.roi_proc.last_expressions)
            except queue.Full:
                pass

            if roi is not None:
                try:
                    self.roi_q.put_nowait(roi)
                except queue.Full:
                    try:
                        self.roi_q.get_nowait()
                        self.roi_q.put_nowait(roi)
                    except queue.Empty:
                        pass
            elif self._use_mock_camera or self.infer.mock_mode:
                mock_roi = np.random.rand(96, 96, 1).astype(np.float32)
                try:
                    self.roi_q.put_nowait(mock_roi)
                except queue.Full:
                    try:
                        self.roi_q.get_nowait()
                        self.roi_q.put_nowait(mock_roi)
                    except queue.Empty:
                        pass

    def _inference_loop(self):
        chunk_buffer = deque(maxlen=self.chunk_size)
        while self.running:
            try:
                roi = self.roi_q.get(timeout=0.1)
                chunk_buffer.append(roi)

                if len(chunk_buffer) == self.chunk_size:
                    t0 = time.perf_counter()
                    chunk = np.array(chunk_buffer)
                    logits = self.infer.run(chunk)

                    if logits is not None:
                        text, conf = self.decoder.decode(logits)
                        latency = time.perf_counter() - t0
                        metrics = self.profiler.log(latency)
                        self.subtitle_ready.emit(text, conf)
                        self.metrics_ready.emit(metrics)

                        # WiFi stream modunda sonucu Pi'ye geri gönder
                        if (self.mode == "stream" and self._stream_server
                                and text.strip()):
                            expr = self.roi_proc.last_expressions
                            dominant = ""
                            if expr:
                                dominant = expr.get("dominant", "")
                                self._stream_server.send_expression(dominant)
                            self._stream_server.send_subtitle(
                                text, conf, dominant
                            )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference thread hatası: {e}", exc_info=True)
                chunk_buffer.clear()
                time.sleep(0.1)  # Hata sonrası throttle

    def _draw_hud(self, annotated_frame: np.ndarray):
        """Foveated HUD overlay çiz — _capture_roi_loop'tan çıkarılmış ortak metot."""
        if self.roi_proc.last_bbox is None:
            return

        x1, y1, x2, y2 = self.roi_proc.last_bbox
        expr = self.roi_proc.last_expressions
        dominant = expr.get("dominant", "Nötr") if expr else "Nötr"
        hud_color = self._EMOTION_BGR.get(dominant, (255, 136, 68))

        # Foveated köşe işaretleri
        corner_len = max(12, (x2 - x1) // 5)
        thickness = 2
        cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), hud_color, thickness)
        cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), hud_color, thickness)
        cv2.line(annotated_frame, (x2, y1), (x2 - corner_len, y1), hud_color, thickness)
        cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_len), hud_color, thickness)
        cv2.line(annotated_frame, (x1, y2), (x1 + corner_len, y2), hud_color, thickness)
        cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_len), hud_color, thickness)
        cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), hud_color, thickness)
        cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), hud_color, thickness)

        # Dudak mesh
        lip_pts = self.roi_proc.last_landmarks
        if lip_pts and len(lip_pts) > 2:
            pts = np.array(lip_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], isClosed=True,
                          color=hud_color, thickness=1, lineType=cv2.LINE_AA)

        # HUD etiket
        if expr:
            conf = expr.get("confidence", 0.0)
            labels_map = {"Gülümseme": "MUTLU", "Kaş Çatma": "KIZGIN",
                          "Şaşırma": "SASIRMIS", "Nötr": "NOTR"}
            lbl_text = f"{labels_map.get(dominant, 'NOTR')} %{int(conf * 100)}"

            kinematic = expr.get("kinematic", {})
            micro = kinematic.get("micro_expression")
            if micro:
                lbl_text += f" [{micro}]"
                hud_color = (255, 0, 255)

            if kinematic.get("is_duchenne", False):
                lbl_text += " [D]"

            label_y = max(y1 - 10, 25)
            (w_text, h_text), _ = cv2.getTextSize(
                lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated_frame,
                          (x1, label_y - h_text - 6),
                          (x1 + w_text + 12, label_y + 4),
                          (26, 14, 10), -1)
            cv2.rectangle(annotated_frame,
                          (x1, label_y - h_text - 6),
                          (x1 + w_text + 12, label_y + 4),
                          hud_color, 1)
            cv2.putText(annotated_frame, lbl_text,
                        (x1 + 6, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        hud_color, 1, cv2.LINE_AA)

            cognitive = expr.get("cognitive", {})
            fatigue = cognitive.get("fatigue_level", "")
            ear = cognitive.get("ear", 0.0)
            if fatigue:
                cog_text = f"EAR:{ear:.2f} [{fatigue}]"
                cog_y = y2 + 18
                cog_color = ((0, 197, 34) if fatigue == "Optimal"
                             else (0, 179, 234) if fatigue == "Normal"
                             else (68, 68, 239))
                cv2.putText(annotated_frame, cog_text,
                            (x1, cog_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, cog_color, 1, cv2.LINE_AA)
