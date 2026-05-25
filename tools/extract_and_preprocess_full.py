"""
tools/extract_and_preprocess_full.py
────────────────────────────────────
Mendeley FULL dataset: dış ZIP → iç ZIP'ler → frame'ler → NPY chunks.

Tek komutla tüm pipeline:
1. data/raw/*.zip içindeki FULL_*.zip'leri aç
2. Her kelime/klip'i data/raw/mendeley_full/ altına çıkar
3. ROI preprocessing → data/processed/ altına [T,96,96,1] float32 kaydeder
4. labels.json günceller

Kullanım:
    python tools/extract_and_preprocess_full.py
    python tools/extract_and_preprocess_full.py --max-frames 30 --roi-size 96 --workers 4
"""

import os
import sys
import io
import json
import time
import zipfile
import argparse
import logging
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── MediaPipe graceful import ─────────────────────────────────────────────────
try:
    import mediapipe as mp
    _face_mesh_cls = mp.solutions.face_mesh.FaceMesh
    _USE_MP = True
    log.info("MediaPipe FaceMesh aktif.")
except (ImportError, AttributeError):
    _USE_MP = False
    log.warning("MediaPipe yok — merkez kırpma (fallback) aktif.")

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
]

# ── ROI Fonksiyonları ─────────────────────────────────────────────────────────

def _roi_mediapipe(frame_bgr, roi_size):
    try:
        with _face_mesh_cls(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as fm:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                xs = [lm[i].x * w for i in LIP_LANDMARKS]
                ys = [lm[i].y * h for i in LIP_LANDMARKS]
                margin = 0.25
                dx = (max(xs) - min(xs)) * margin
                dy = (max(ys) - min(ys)) * margin
                x1 = max(0, int(min(xs) - dx))
                y1 = max(0, int(min(ys) - dy))
                x2 = min(w, int(max(xs) + dx))
                y2 = min(h, int(max(ys) + dy))
                roi = frame_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    return cv2.resize(roi, (roi_size, roi_size))
    except Exception:
        pass
    return None


def _roi_fallback(frame_bgr, roi_size):
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.55):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
    if roi.size == 0:
        roi = frame_bgr
    return cv2.resize(roi, (roi_size, roi_size))


def frame_to_roi(frame_bgr, roi_size):
    roi = _roi_mediapipe(frame_bgr, roi_size) if _USE_MP else None
    if roi is None:
        roi = _roi_fallback(frame_bgr, roi_size)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray[:, :, np.newaxis]  # [H, W, 1]

# ── Klip işleyici (tek klip → NPY) ───────────────────────────────────────────

def process_clip_dir(clip_dir, out_path, roi_size, max_frames):
    """Klasördeki JPG'leri oku → ROI → padding → NPY kaydet."""
    jpgs = sorted(
        f for f in os.listdir(clip_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not jpgs:
        return False, "frame yok"

    rois = []
    for fname in jpgs:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is not None:
            rois.append(frame_to_roi(img, roi_size))

    if not rois:
        return False, "okuma hatasi"

    # Padding / kırpma
    blank = np.zeros((roi_size, roi_size, 1), dtype=np.float32)
    if len(rois) >= max_frames:
        rois = rois[:max_frames]
    else:
        rois += [blank] * (max_frames - len(rois))

    chunk = np.stack(rois, axis=0)  # [T, H, W, 1]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, chunk)
    return True, len(jpgs)

# ── Step 1: İç ZIP'leri çıkar ────────────────────────────────────────────────

def extract_full_zips(outer_zip_path, extract_root):
    """Dış ZIP içindeki FULL inner ZIP'leri çıkar → extract_root/kelime/clip/frame.jpg"""
    os.makedirs(extract_root, exist_ok=True)
    extracted = {}  # word → clip_count

    with zipfile.ZipFile(outer_zip_path, "r") as outer:
        full_zips = [n for n in outer.namelist() if "FULL" in n and n.endswith(".zip")]
        log.info(f"Cikartilacak {len(full_zips)} ic ZIP bulundu.")

        for i, fz_path in enumerate(sorted(full_zips), 1):
            word = Path(fz_path).stem  # merhaba, selam, ...
            word_dir = os.path.join(extract_root, word)
            os.makedirs(word_dir, exist_ok=True)

            log.info(f"[{i}/{len(full_zips)}] Aciliyor: {word}.zip")
            with outer.open(fz_path) as f:
                inner_bytes = io.BytesIO(f.read())
                with zipfile.ZipFile(inner_bytes) as inner:
                    members = inner.namelist()
                    clips_done = set()
                    for member in members:
                        if not member.lower().endswith((".jpg", ".jpeg")):
                            continue
                        parts = member.strip("/").split("/")
                        # Beklenen: <word>/<clip_id>/<frame.jpg>
                        if len(parts) < 3:
                            continue
                        clip_id = parts[1]
                        frame_name = parts[2]
                        clip_out = os.path.join(word_dir, clip_id)
                        os.makedirs(clip_out, exist_ok=True)
                        out_file = os.path.join(clip_out, frame_name)
                        if not os.path.exists(out_file):
                            data = inner.read(member)
                            with open(out_file, "wb") as fp:
                                fp.write(data)
                        clips_done.add(clip_id)
                    extracted[word] = len(clips_done)
                    log.info(f"  -> {word}: {len(clips_done)} klip cikartildi")
    return extracted

# ── Step 2: Preprocessing ────────────────────────────────────────────────────

def preprocess_all(extract_root, processed_root, roi_size, max_frames):
    """extract_root altındaki tüm klipleri işle → NPY."""
    os.makedirs(processed_root, exist_ok=True)
    labels = {}
    stats = {"ok": 0, "fail": 0, "skip": 0}
    t0 = time.time()

    words = sorted(
        d for d in os.listdir(extract_root)
        if os.path.isdir(os.path.join(extract_root, d))
    )
    log.info(f"\nOnisleme basliyor: {len(words)} kelime")

    for wi, word in enumerate(words, 1):
        word_in = os.path.join(extract_root, word)
        word_out = os.path.join(processed_root, word)
        os.makedirs(word_out, exist_ok=True)

        clips = sorted(
            d for d in os.listdir(word_in)
            if os.path.isdir(os.path.join(word_in, d))
        )
        log.info(f"[{wi}/{len(words)}] {word}: {len(clips)} klip isleniyor...")

        word_ok = 0
        for clip_id in clips:
            clip_dir = os.path.join(word_in, clip_id)
            out_fname = f"{word}_{clip_id}.npy"
            out_path = os.path.join(word_out, out_fname)
            rel_path = f"{word}/{out_fname}"

            if os.path.exists(out_path):
                labels[rel_path] = word
                stats["skip"] += 1
                word_ok += 1
                continue

            ok, info = process_clip_dir(clip_dir, out_path, roi_size, max_frames)
            if ok:
                labels[rel_path] = word
                stats["ok"] += 1
                word_ok += 1
            else:
                log.warning(f"  FAIL {rel_path}: {info}")
                stats["fail"] += 1

        log.info(f"  {word}: {word_ok}/{len(clips)} basarili")

    # labels.json yaz
    labels_path = os.path.join(processed_root, "labels.json")
    existing = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(labels)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n{'='*55}")
    log.info(f"Onisleme Tamamlandi ({elapsed:.1f}s)")
    log.info(f"  Basarili : {stats['ok']}")
    log.info(f"  Atlandi  : {stats['skip']} (zaten mevcut)")
    log.info(f"  Basarisiz: {stats['fail']}")
    log.info(f"  Labels   : {labels_path} ({len(existing)} kayit)")
    log.info(f"{'='*55}\n")
    return existing

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mendeley FULL Dataset: cikart + onisle"
    )
    parser.add_argument(
        "--zip", default=r"data/raw/Visual Lip Reading Dataset in Turkish.zip",
        help="Dis ZIP dosyasi yolu"
    )
    parser.add_argument(
        "--extract-dir", default="data/raw/mendeley_full",
        help="Cikarma hedef klasoru"
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Onislenms NPY ciktisi"
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--roi-size",   type=int, default=96)
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Cikarma adimini atla (zaten yapildi)"
    )
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("Blind Eye | Mendeley FULL Dataset Pipeline")
    log.info(f"ZIP    : {args.zip}")
    log.info(f"Hedef  : {args.extract_dir}")
    log.info(f"Cikti  : {args.output}")
    log.info(f"Frames : max={args.max_frames}, roi={args.roi_size}x{args.roi_size}")
    log.info("=" * 55)

    # Step 1: Extract
    if not args.skip_extract:
        log.info("\n[ADIM 1/2] IC ZIP'LER CIKARTILIYOR...")
        extracted = extract_full_zips(args.zip, args.extract_dir)
        total_clips = sum(extracted.values())
        log.info(f"Cikartma tamamlandi: {total_clips} toplam klip")
    else:
        log.info("[ADIM 1/2] Atlandi (--skip-extract)")

    # Step 2: Preprocess
    log.info("\n[ADIM 2/2] ONISLEME BASLIYOR...")
    labels = preprocess_all(
        args.extract_dir, args.output,
        args.roi_size, args.max_frames
    )

    log.info(f"Pipeline tamamlandi. Toplam {len(labels)} ornek hazir.")
    log.info("Sonraki adim: python tools/augment.py --input data/processed --output data/augmented --factor 3")


if __name__ == "__main__":
    main()
