import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime yüklü değil — MockInference kullanılacak.")


class InferenceEngine:
    """ONNX model yükleme + chunk çıkarımı. Model yoksa mock mod.

    num_classes configs/vocab.json'dan okunur.
    """

    def __init__(self, model_path: str, chunk_size: int = 6,
                 providers: list = None):
        self.chunk_size = chunk_size
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = None
        self.input_name = None
        self.mock_mode = True
        self.num_classes = self._load_num_classes()

        if ORT_AVAILABLE:
            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=self.providers
                )
                self.input_name = self.session.get_inputs()[0].name
                self.mock_mode = False
                logger.info(f"ONNX modeli yüklendi: {model_path}")
            except Exception as e:
                logger.warning(f"Model yüklenemedi, mock mod aktif: {e}")

        if self.mock_mode:
            logger.info("MockInference aktif — dummy çıktı üretilecek.")

    def _load_num_classes(self) -> int:
        """configs/vocab.json'dan num_classes değerini okur."""
        try:
            from .decoder import load_vocab
            _, _, num_classes = load_vocab()
            return num_classes
        except Exception:
            return 31  # Varsayılan: 29 harf + blank + space

    def run(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """ONNX model inference.

        Args:
            chunk: [T, H, W, C] boyutlu ROI chunk'ı

        Returns:
            [1, T, num_classes] boyutlu logits veya None
        """
        if self.mock_mode:
            return self._mock_run(chunk)

        try:
            # [T, H, W, C] → [1, T, H, W, C] batch dim ekle
            input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            outputs = self.session.run(
                None, {self.input_name: input_data}
            )
            return outputs[0]
        except Exception as e:
            logger.error(f"Inference hatası: {e}")
            return None

    def _mock_run(self, chunk: np.ndarray) -> np.ndarray:
        """Mock inference — gerçek model yokken dummy logits üretir."""
        import time
        time.sleep(0.03)
        t = chunk.shape[0] if chunk is not None else self.chunk_size
        logits = np.random.randn(1, t, self.num_classes).astype(np.float32)
        return logits
