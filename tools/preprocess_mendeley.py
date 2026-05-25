"""
tools/preprocess_mendeley.py
────────────────────────────
Mendeley Turkish Lip Reading Dataset işleyicisi.

Dataset yapısı:
    data/raw/mendeley/<kelime>/<clip_id>/<01.jpg ... NN.jpg>

Çıktı:
    data/processed/<kelime>/<kelime>_<clip_id>.npy  →  [T, 96, 96, 1] float32
    data/processed/labels.json

Kullanım:
    python tools/preprocess_mendeley.py
    python tools/preprocess_mendeley.py --input data/raw/mendeley --output data/processed --max-frames 30 --roi-size 96
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

# MediaPipe graceful import
try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
    _USE_MP = True
    print("[INFO] MediaPipe FaceMesh aktif.")
except (ImportError, AttributeError):
    _USE_MP = False
    print("[WARN] MediaPipe kullanılamıyor — merkez kırpma (fallback) aktif.")


# ── Dudak landmark indisleri (MediaPipe FaceMesh 468 nokta) ──────────────────
LIP_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIP_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIP_LANDMARKS = list(set(LIP_UPPER + LIP_LOWER))


def extract_lip_roi_mp(frame_bgr: np.ndarray, roi_size: int, margin: float = 0.25):
    """MediaPipe ile dudak ROI çıkar."""
    try:
        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        ) as face_mesh:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                xs = [lm[i].x * w for i in LIP_LANDMARKS]
                ys = [lm[i].y * h for i in LIP_LANDMARKS]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                dx = (x_max - x_min) * margin
                dy = (y_max - y_min) * margin
                x1 = max(0, int(x_min - dx))
                y1 = max(0, int(y_min - dy))
                x2 = min(w, int(x_max + dx))
                y2 = min(h, int(y_max + dy))
                roi = frame_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.resize(roi, (roi_size, roi_size))
                    return roi
    except Exception:
        pass
    return None


def extract_lip_roi_fallback(frame_bgr: np.ndarray, roi_size: int):
    """Sabit koordinatlarla merkez kırpma (fallback)."""
    h, w = frame_bgr.shape[:2]
    y1, y2 = int(h * 0.55), int(h * 0.85)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = frame_bgr
    return cv2.resize(roi, (roi_size, roi_size))


def load_frame_sequence(clip_dir: str):
    """Klip klasöründen sıralı JPG frame'leri yükle."""
    frames = []
    jpg_files = sorted(
        f for f in os.listdir(clip_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    for fname in jpg_files:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is not None:
            frames.append(img)
    return frames


def process_clip(clip_dir: str, roi_size: int, max_frames: int):
    """
    Tek bir klip klasörünü işle → [max_frames, roi_size, roi_size, 1] float32.
    """
    raw_frames = load_frame_sequence(clip_dir)
    if not raw_frames:
        return None

    rois = []
    for frame in raw_frames:
        roi = None
        if _USE_MP:
            roi = extract_lip_roi_mp(frame, roi_size)
        if roi is None:
            roi = extract_lip_roi_fallback(frame, roi_size)

        # Grayscale → normalize
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # [H, W]
        gray = gray.astype(np.float32) / 255.0
        gray = gray[:, :, np.newaxis]  # [H, W, 1]
        rois.append(gray)

    # Padding / kırpma → sabit uzunluk
    T = len(rois)
    if T >= max_frames:
        rois = rois[:max_frames]
    else:
        pad_frame = np.zeros((roi_size, roi_size, 1), dtype=np.float32)
        rois += [pad_frame] * (max_frames - T)

    return np.stack(rois, axis=0)  # [max_frames, roi_size, roi_size, 1]


def main():
    parser = argparse.ArgumentParser(description="Mendeley Türkçe Dudak Okuma Dataset Preprocessor")
    parser.add_argument("--input",  default="data/raw/mendeley", help="Frame dizisi giriş klasörü")
    parser.add_argument("--output", default="data/processed",    help="Çıktı klasörü")
    parser.add_argument("--max-frames", type=int, default=30,    help="Sabit çıktı uzunluğu")
    parser.add_argument("--roi-size",   type=int, default=96,    help="ROI boyutu (kare)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    labels = {}
    total_ok = 0
    total_fail = 0

    words = sorted(d for d in os.listdir(args.input)
                   if os.path.isdir(os.path.join(args.input, d)))

    print(f"\n[INFO] {len(words)} kelime bulundu: {words}")
    print(f"[INFO] ROI boyutu: {args.roi_size}×{args.roi_size}, max frame: {args.max_frames}\n")

    for word in words:
        word_in  = os.path.join(args.input,  word)
        word_out = os.path.join(args.output, word)
        os.makedirs(word_out, exist_ok=True)

        clips = sorted(d for d in os.listdir(word_in)
                       if os.path.isdir(os.path.join(word_in, d)))

        for clip_id in clips:
            clip_dir  = os.path.join(word_in, clip_id)
            out_fname = f"{word}_{clip_id}.npy"
            out_path  = os.path.join(word_out, out_fname)
            rel_path  = os.path.join(word, out_fname).replace("\\", "/")

            frames = load_frame_sequence(clip_dir)
            if not frames:
                print(f"  [SKIP] {word}/{clip_id} — frame yok")
                total_fail += 1
                continue

            chunk = process_clip(clip_dir, args.roi_size, args.max_frames)
            if chunk is None:
                print(f"  [FAIL] {word}/{clip_id}")
                total_fail += 1
                continue

            np.save(out_path, chunk)
            labels[rel_path] = word
            total_ok += 1
            print(f"  [OK]  {rel_path}  shape={chunk.shape}  ({len(frames)} ham frame)")

    # labels.json kaydet
    labels_path = os.path.join(args.output, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Başarılı : {total_ok}")
    print(f"❌ Başarısız: {total_fail}")
    print(f"📁 Çıktı   : {args.output}/")
    print(f"🏷️  Labels  : {labels_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
