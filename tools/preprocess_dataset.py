"""
tools/preprocess_dataset.py
Mendeley Turkish Lip Reading Dataset → ROI Chunk Pipeline

Kullanım:
    python tools/preprocess_dataset.py --input data/raw --output data/processed

Beklenen klasör yapısı (input):
    data/raw/
    ├── merhaba/
    │   ├── merhaba_01.mp4
    │   ├── merhaba_02.mp4
    │   └── ...
    ├── evet/
    │   ├── evet_01.mp4
    │   └── ...
    └── ...

Çıktı yapısı (output):
    data/processed/
    ├── merhaba/
    │   ├── merhaba_01.npy   # [T, 96, 96, 1] float32
    │   └── ...
    └── labels.json          # {"merhaba_01": "merhaba", ...}
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MediaPipe import (opsiyonel)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    MP_AVAILABLE = False
    logger.warning("MediaPipe yüklü değil — basit ROI kırpma kullanılacak.")


# ═══════════════════════════════════════════════════════════════
#  1. ROI ÇIKARMA
# ═══════════════════════════════════════════════════════════════

class ROIExtractor:
    """MediaPipe FaceMesh ile dudak ROI çıkarma.
    MediaPipe yoksa basit cascade + merkez kırpma kullanır.
    """

    # MediaPipe dudak landmark indeksleri
    LIP_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
        415, 308,
    ]

    def __init__(self, roi_size: Tuple[int, int] = (96, 96), margin: float = 0.3):
        self.roi_size = roi_size
        self.margin = margin
        self.face_mesh = None

        if MP_AVAILABLE:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                )
            except (AttributeError, Exception) as e:
                logger.warning(f"MediaPipe FaceMesh başlatılamadı: {e}")
                self.face_mesh = None

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Frame'den dudak ROI çıkarır.

        Returns:
            [H, W, 1] float32 normalize edilmiş ROI veya None
        """
        if self.face_mesh is not None:
            return self._extract_mediapipe(frame)
        return self._extract_simple(frame)

    def _extract_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe FaceMesh ile hassas dudak ROI."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Dudak bounding box hesapla
        lip_pts = []
        for idx in self.LIP_INDICES:
            lm = landmarks.landmark[idx]
            lip_pts.append((int(lm.x * w), int(lm.y * h)))

        xs = [p[0] for p in lip_pts]
        ys = [p[1] for p in lip_pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Margin ekle
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * self.margin))
        y1 = max(0, int(y1 - bh * self.margin))
        x2 = min(w, int(x2 + bw * self.margin))
        y2 = min(h, int(y2 + bh * self.margin))

        roi = frame[y1:y2, x1:x2]
        return self._normalize(roi)

    def _extract_simple(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Basit merkez kırpma (fallback)."""
        h, w = frame.shape[:2]
        # Alt ortadan %30 bölge al (dudak yaklaşık konumu)
        y1 = int(h * 0.55)
        y2 = int(h * 0.85)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        roi = frame[y1:y2, x1:x2]
        return self._normalize(roi)

    def _normalize(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """ROI'yi gri tonlama, resize, normalize et."""
        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        resized = cv2.resize(gray, self.roi_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[..., np.newaxis]  # [H, W, 1]

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()


# ═══════════════════════════════════════════════════════════════
#  2. VİDEO İŞLEME
# ═══════════════════════════════════════════════════════════════

def process_video(
    video_path: str,
    extractor: ROIExtractor,
    max_frames: int = 75,
    target_fps: int = 25,
) -> Optional[np.ndarray]:
    """Videoyu ROI frame dizisine çevirir.

    Returns:
        [T, H, W, 1] float32 ndarray veya None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Video açılamadı: {video_path}")
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = max(1, int(src_fps / target_fps))

    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            roi = extractor.extract(frame)
            if roi is not None:
                frames.append(roi)

        frame_idx += 1

    cap.release()

    if len(frames) < 3:
        logger.warning(f"Yetersiz frame ({len(frames)}): {video_path}")
        return None

    # Padding: eksik frame'leri sıfırla doldur
    if len(frames) < max_frames:
        pad = np.zeros_like(frames[0])
        while len(frames) < max_frames:
            frames.append(pad)

    return np.array(frames[:max_frames], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  3. BATCH İŞLEME
# ═══════════════════════════════════════════════════════════════

def process_dataset(
    input_dir: str,
    output_dir: str,
    roi_size: Tuple[int, int] = (96, 96),
    max_frames: int = 75,
) -> dict:
    """Tüm veri setini işler.

    Returns:
        labels dict: {sample_id: label}
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    extractor = ROIExtractor(roi_size=roi_size)
    labels = {}
    stats = {"total": 0, "success": 0, "failed": 0}

    # Her alt klasör bir etiket
    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if not class_dirs:
        logger.error(f"Veri klasörleri bulunamadı: {input_dir}")
        return labels

    logger.info(f"📂 {len(class_dirs)} sınıf bulundu: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        label = class_dir.name
        out_class_dir = output_path / label
        out_class_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        ])

        logger.info(f"  📁 {label}: {len(video_files)} video")

        for vf in video_files:
            stats["total"] += 1
            sample_id = vf.stem

            chunks = process_video(str(vf), extractor, max_frames=max_frames)
            if chunks is None:
                stats["failed"] += 1
                continue

            # .npy olarak kaydet
            out_file = out_class_dir / f"{sample_id}.npy"
            np.save(str(out_file), chunks)
            labels[sample_id] = label
            stats["success"] += 1

    extractor.close()

    # Labels kaydet
    labels_path = output_path / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    logger.info(f"\n📊 İşlem Tamamlandı:")
    logger.info(f"  Toplam: {stats['total']}")
    logger.info(f"  Başarılı: {stats['success']}")
    logger.info(f"  Başarısız: {stats['failed']}")
    logger.info(f"  Etiketler: {labels_path}")

    return labels


# ═══════════════════════════════════════════════════════════════
#  4. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mendeley Turkish Lip Reading → ROI Chunk Pipeline"
    )
    parser.add_argument("--input", default="data/raw",
                        help="Ham video klasörü (alt klasörler = etiketler)")
    parser.add_argument("--output", default="data/processed",
                        help="İşlenmiş .npy çıktı klasörü")
    parser.add_argument("--roi-size", type=int, nargs=2, default=[96, 96],
                        help="ROI boyutu [genişlik yükseklik]")
    parser.add_argument("--max-frames", type=int, default=75,
                        help="Video başına maksimum frame sayısı")

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        roi_size=tuple(args.roi_size),
        max_frames=args.max_frames,
    )
