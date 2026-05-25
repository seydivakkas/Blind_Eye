"""
pc/vsr_engine.py
════════════════
Lightweight VSR — ResNet18 + DC-TCN + CTC — ONNX GPU Inference

Mevcut student_fp32.onnx modelini RTX 4070 GPU üzerinde çalıştırır.
Model giriş/çıkış:
    Input : [1, T=30, 96, 96, 1]  — T frame'lik grayscale ROI chunk
    Output: [1, T=30, 31]         — CTC logits (31 = 30 TR karakter + blank)

Mimari (ONNX içinde):
    3D-Conv (spatio-temporal) → ResNet18 (frame-wise spatial)
    → DC-TCN (densely connected temporal) → Linear → CTC logits

Bu modül sadece raw logits üretir — decode işlemi fusion_decoder.py'de yapılır.

Kullanım:
    engine = VSREngine(model_path="models/student_fp32.onnx")
    logits = engine.infer(roi_batch)    # [1, T, 31] np.ndarray
    engine.close()

Referans:
    - Ma et al., "Lip-reading with Densely Connected TCNs" (WACV 2021)
    - ONNX Runtime CUDAExecutionProvider documentation
"""

import logging
import os
import time
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("onnxruntime yüklü değil — VSREngine mock modda çalışacak.")


class VSREngine:
    """ResNet18 + DC-TCN + CTC ONNX GPU inference motoru.

    Parameters
    ----------
    model_path : str
        ONNX model dosya yolu.
    chunk_size : int
        Model giriş temporal uzunluğu (T). Varsayılan 30.
    providers : list[str] | None
        ONNX Runtime execution provider'ları. None ise otomatik seçim.
        RTX 4070 için: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    num_threads : int
        CPU inference için thread sayısı (GPU yoksa).
    warmup : bool
        True ise ilk inference'ta dummy veri ile ısınma yapar.
    """

    def __init__(
        self,
        model_path: str = "models/student_fp32.onnx",
        chunk_size: int = 30,
        providers: Optional[List[str]] = None,
        num_threads: int = 4,
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.num_classes = 31  # 30 TR karakter + blank

        self._session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self.mock_mode = True
        self.active_provider = "MockProvider"

        # İstatistikler
        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0

        if not _ORT_AVAILABLE:
            logger.warning("ONNX Runtime yok — mock mod aktif")
            return

        if not os.path.exists(model_path):
            logger.warning(f"Model bulunamadı: {model_path} — mock mod aktif")
            return

        # ── ONNX Runtime Session oluştur ──
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = num_threads

            # Provider seçimi: GPU > CPU
            if providers is None:
                available = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "TensorrtExecutionProvider" in available:
                    providers.insert(0, "TensorrtExecutionProvider")
                providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )

            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self.active_provider = self._session.get_providers()[0]
            self.mock_mode = False

            # Model bilgilerini logla
            inp = self._session.get_inputs()[0]
            out = self._session.get_outputs()[0]
            logger.info(
                f"✅ VSREngine hazır: {model_path}\n"
                f"   Provider : {self.active_provider}\n"
                f"   Input    : {inp.name} {inp.shape}\n"
                f"   Output   : {out.name} {out.shape}"
            )

            # Modelden chunk_size ve num_classes güncelle
            if inp.shape and inp.shape[1]:
                self.chunk_size = inp.shape[1]
            if out.shape and out.shape[2]:
                self.num_classes = out.shape[2]

            # Warm-up
            if warmup:
                self._warmup()

        except Exception as e:
            logger.error(f"ONNX model yüklenemedi: {e} — mock mod aktif")
            self.mock_mode = True

    # ──────────── PUBLIC API ────────────

    def infer(self, roi_batch: np.ndarray) -> Optional[np.ndarray]:
        """ROI chunk'ından CTC logits üret.

        Parameters
        ----------
        roi_batch : np.ndarray
            [T, 96, 96, 1] float32 — preprocessed ROI frame dizisi.
            T, self.chunk_size ile eşleşmeli.

        Returns
        -------
        np.ndarray | None
            [1, T, num_classes] float32 logits. Hata durumunda None.
        """
        if self.mock_mode:
            return self._mock_infer(roi_batch)

        try:
            # Batch dimension ekle: [T, H, W, C] → [1, T, H, W, C]
            if roi_batch.ndim == 4:
                input_data = np.expand_dims(roi_batch, axis=0).astype(np.float32)
            elif roi_batch.ndim == 5:
                input_data = roi_batch.astype(np.float32)
            else:
                logger.error(f"Beklenmeyen input shape: {roi_batch.shape}")
                return None

            t0 = time.perf_counter()
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: input_data},
            )
            self.last_latency_ms = (time.perf_counter() - t0) * 1000
            self.total_latency_ms += self.last_latency_ms
            self.inference_count += 1

            return outputs[0]  # [1, T, num_classes]

        except Exception as e:
            logger.error(f"Inference hatası: {e}")
            return None

    def get_avg_latency_ms(self) -> float:
        """Ortalama inference süresi (ms)."""
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count

    def close(self):
        """Kaynakları serbest bırak."""
        self._session = None
        logger.info(
            f"VSREngine kapatıldı — {self.inference_count} inference, "
            f"avg {self.get_avg_latency_ms():.1f}ms"
        )

    # ──────────── INTERNAL ────────────

    def _warmup(self):
        """Dummy veri ile GPU ısınması."""
        logger.info("VSREngine warm-up başlıyor...")
        dummy = np.random.rand(
            1, self.chunk_size, 96, 96, 1
        ).astype(np.float32)
        try:
            t0 = time.perf_counter()
            self._session.run(
                [self._output_name],
                {self._input_name: dummy},
            )
            warmup_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"✅ Warm-up tamamlandı: {warmup_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"Warm-up hatası (devam ediliyor): {e}")

    def _mock_infer(self, roi_batch: np.ndarray) -> np.ndarray:
        """Mock inference — model yokken dummy logits üretir."""
        time.sleep(0.01)  # Simüle latency
        if roi_batch is not None and roi_batch.ndim >= 4:
            t = roi_batch.shape[0] if roi_batch.ndim == 4 else roi_batch.shape[1]
        else:
            t = self.chunk_size
        self.last_latency_ms = 10.0
        self.inference_count += 1
        return np.random.randn(1, t, self.num_classes).astype(np.float32)
