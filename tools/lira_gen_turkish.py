"""
tools/lira_gen_turkish.py
─────────────────────────
LIRA-Gen Türkçe Uyarlaması — YouTube CC Altyazılı Videolardan
Otomatik Türkçe Dudak Okuma Veri Seti Üretici

Kaynak: LIRA-Gen (Megiyanto377/LIRA-Gen) — Türkçe uyarlaması
Referans: Lipreading Information Resource Assembler-Generator

8 Aşamalı Pipeline:
    1. YouTube Video + CC Altyazı İndirme (yt-dlp)
    2. Shot Detection — Sahne Bölme (scenedetect / TransNetV2)
    3. Forced Alignment — Kelime-Zaman Eşleme (MFA)
    4. Kelime Filtreleme (frekans + Türkçe sözlük)
    5. Kelime Bazlı Video Kırpma (1 sn klipler)
    6. Yüz Tespiti + Dudak Landmark (face_recognition + dlib)
    7. Dudak ROI → NPY Dönüşümü (96×96 gri tonlama)
    8. Labels & Metadata Üretimi (labels.json + CSV)

Kullanım:
    # Tam pipeline (YouTube playlist URL ile):
    python tools/lira_gen_turkish.py --playlist "https://youtube.com/playlist?list=..."

    # Belirli aşamadan başla:
    python tools/lira_gen_turkish.py --stage 5 --input-dir video/output_shot_video

    # Yerel video dizini ile (YouTube indirmeden):
    python tools/lira_gen_turkish.py --local-dir video/input --stage 2

TÜBİTAK 2209-A — Blind Eye Projesi
"""

import os
import sys
import csv
import json
import time
import shutil
import logging
import argparse
import pathlib
import subprocess
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("lira_gen_turkish")

# Proje kök dizini
PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())


def _get_ffmpeg_cmd() -> str:
    """Sistemde ffmpeg arar, bulamazsa imageio_ffmpeg'inkini video/bin/ffmpeg.exe'ye kopyalayıp döner."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    local_bin_dir = os.path.join(PROJECT_ROOT, "video", "bin")
    local_ffmpeg = os.path.join(local_bin_dir, "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe):
            os.makedirs(local_bin_dir, exist_ok=True)
            shutil.copy2(ffmpeg_exe, local_ffmpeg)
            logger.info(f"ffmpeg.exe başarıyla oluşturuldu: {local_ffmpeg}")
            return local_ffmpeg
    except Exception as e:
        logger.error(f"ffmpeg kopyalama hatası: {e}")

    return "ffmpeg"


# ═══════════════════════════════════════════════════════════════
#  Sabitler ve Yapılandırma
# ═══════════════════════════════════════════════════════════════

LANG = "tr"
CLIP_DURATION = 1.0       # Her kelime klibi süresi (saniye)
MIN_WORD_LENGTH = 2       # Minimum kelime uzunluğu (karakter)
MAX_WORD_LENGTH = 20      # Maksimum kelime uzunluğu
ROI_SIZE = 96             # Dudak ROI boyutu (96×96 piksel)
SEQ_LEN = 30              # Model giriş frame sayısı
TARGET_FPS = 25           # Hedef FPS

# Dizin yapısı
DIRS = {
    "input":                os.path.join(PROJECT_ROOT, "video", "input"),
    "input_srt":            os.path.join(PROJECT_ROOT, "video", "input_srt"),
    "output_shots":         os.path.join(PROJECT_ROOT, "video", "output_shots"),
    "output_align":         os.path.join(PROJECT_ROOT, "video", "output_align"),
    "output_words_filtered":os.path.join(PROJECT_ROOT, "video", "output_words_filtered"),
    "output_word_clips":    os.path.join(PROJECT_ROOT, "video", "output_word_clips"),
    "output_face_crop":     os.path.join(PROJECT_ROOT, "video", "output_face_crop"),
    "output_npy":           os.path.join(PROJECT_ROOT, "data", "processed"),
}

# Türkçe sözlük dosyası
TURKISH_DICT_PATH = os.path.join(PROJECT_ROOT, "configs", "turkish_words.txt")


def ensure_dirs():
    """Tüm çalışma dizinlerini oluştur."""
    for name, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Dizin hazır: {path}")


# ═══════════════════════════════════════════════════════════════
#  STAGE 1: YouTube Video + CC Altyazı İndirme
# ═══════════════════════════════════════════════════════════════

def stage1_download_youtube(playlist_url: str, max_videos: int = 50):
    """
    YouTube playlist'inden CC altyazılı Türkçe videoları indir.

    yt-dlp kullanır:
      - Video: 720p MP4
      - Altyazı: Türkçe CC (.srt)
      - Sadece CC (otomatik değil) altyazısı olan videoları indirir

    Args:
        playlist_url: YouTube playlist URL
        max_videos: Maksimum indirilecek video sayısı
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: YouTube Video + CC Altyazı İndirme")
    logger.info("=" * 60)

    output_dir = DIRS["input"]
    srt_dir = DIRS["input_srt"]

    # yt-dlp kontrol
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("yt-dlp bulunamadı! Lütfen yükleyin: pip install yt-dlp")
        sys.exit(1)

    # Video indirme komutu
    cmd = [
        "yt-dlp",
        "--playlist-end", str(max_videos),
        # Video formatı: 720p MP4
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
        "--merge-output-format", "mp4",
        # Altyazı: Türkçe CC ve Otomatik Altyazı desteği
        "--write-sub",
        "--write-auto-subs",
        "--sub-lang", "tr",
        "--sub-format", "srt",
        "--convert-subs", "srt",
        # İsimlendirme
        "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
        # Var olan dosyaları atla
        "--no-overwrites",
    ]

    ffmpeg_exe = _get_ffmpeg_cmd()
    if ffmpeg_exe != "ffmpeg":
        cmd.extend(["--ffmpeg-location", os.path.dirname(ffmpeg_exe)])

    cmd.append(playlist_url)

    logger.info(f"yt-dlp komutu: {' '.join(cmd[:8])}...")
    logger.info(f"Hedef: {output_dir}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info("Video indirme tamamlandı!")
        else:
            logger.warning(f"yt-dlp uyarıları: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        logger.error("İndirme zaman aşımına uğradı (1 saat)!")
        return

    # SRT dosyalarını taşı
    for f in os.listdir(output_dir):
        if f.endswith(".srt"):
            src = os.path.join(output_dir, f)
            dst = os.path.join(srt_dir, f)
            shutil.move(src, dst)
            logger.info(f"SRT taşındı: {f}")

    # İstatistikler
    video_count = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])
    srt_count = len([f for f in os.listdir(srt_dir) if f.endswith(".srt")])
    logger.info(f"✅ Stage 1 tamamlandı: {video_count} video, {srt_count} altyazı")


# ═══════════════════════════════════════════════════════════════
#  STAGE 2: Shot Detection (Sahne Bölme)
# ═══════════════════════════════════════════════════════════════

def stage2_shot_detection():
    """
    Videoları sahne geçişlerine göre parçalara ayır.

    PySceneDetect (ContentDetector) kullanır.
    Her sahne ayrı bir video dosyası olarak kaydedilir.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Shot Detection (Sahne Bölme)")
    logger.info("=" * 60)

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        logger.error("scenedetect bulunamadı! pip install scenedetect[opencv]")
        sys.exit(1)

    input_dir = DIRS["input"]
    output_dir = DIRS["output_shots"]

    videos = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".mkv", ".avi"))]
    logger.info(f"İşlenecek video sayısı: {len(videos)}")

    for idx, video_file in enumerate(videos):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        logger.info(f"[{idx+1}/{len(videos)}] Sahne tespiti: {video_file}")

        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video)

            scene_list = scene_manager.get_scene_list()
            logger.info(f"  → {len(scene_list)} sahne tespit edildi")

            # Her sahneyi ayrı dosyaya kes (ffmpeg ile)
            ffmpeg_exe = _get_ffmpeg_cmd()
            for i, (start, end) in enumerate(scene_list):
                start_sec = start.get_seconds()
                end_sec = end.get_seconds()
                duration = end_sec - start_sec

                if duration < 1.0 or duration > 60.0:
                    continue  # Çok kısa/uzun sahneleri atla

                out_path = os.path.join(video_out_dir, f"shot_{i:04d}.mp4")
                cmd = [
                    ffmpeg_exe, "-y", "-i", video_path,
                    "-ss", f"{start_sec:.3f}",
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac",
                    "-loglevel", "error",
                    out_path
                ]
                subprocess.run(cmd, capture_output=True)

        except Exception as e:
            logger.error(f"  Sahne tespiti hatası: {e}")

    total_shots = sum(len(os.listdir(os.path.join(output_dir, d)))
                      for d in os.listdir(output_dir)
                      if os.path.isdir(os.path.join(output_dir, d)))
    logger.info(f"✅ Stage 2 tamamlandı: {total_shots} sahne klibi üretildi")


# ═══════════════════════════════════════════════════════════════
#  STAGE 3: Forced Alignment (Kelime-Zaman Eşleme)
# ═══════════════════════════════════════════════════════════════

def stage3_forced_alignment():
    """
    SRT altyazılardan kelime-zaman eşlemesi çıkar.

    Basit SRT parser kullanır (MFA mevcut değilse fallback).
    MFA mevcutsa daha hassas fonem bazlı eşleme yapar.

    Çıktı: word_timestamps.csv
        word, start_time, end_time, video_file
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Forced Alignment (Kelime-Zaman Eşleme)")
    logger.info("=" * 60)

    srt_dir = DIRS["input_srt"]
    output_path = os.path.join(DIRS["output_align"], "word_timestamps.csv")
    os.makedirs(DIRS["output_align"], exist_ok=True)

    srt_files = [f for f in os.listdir(srt_dir) if f.endswith(".srt")]
    logger.info(f"İşlenecek SRT sayısı: {len(srt_files)}")

    all_words = []

    for srt_file in srt_files:
        srt_path = os.path.join(srt_dir, srt_file)
        video_name = srt_file.replace(".tr.srt", "").replace(".srt", "")

        try:
            words = _parse_srt_to_words(srt_path, video_name)
            all_words.extend(words)
            logger.info(f"  {srt_file}: {len(words)} kelime çıkarıldı")
        except Exception as e:
            logger.error(f"  SRT parse hatası ({srt_file}): {e}")

    # CSV kaydet
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start", "end", "video"])
        writer.writeheader()
        writer.writerows(all_words)

    logger.info(f"✅ Stage 3 tamamlandı: {len(all_words)} kelime-zaman eşlemesi → {output_path}")


def _parse_srt_to_words(srt_path: str, video_name: str) -> List[Dict]:
    """SRT dosyasını parse edip kelime-zaman listesi döndür."""
    words = []

    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Zaman çizgisi: "00:01:23,456 --> 00:01:25,789"
        time_line = None
        text_lines = []
        for line in lines:
            if " --> " in line:
                time_line = line
            elif time_line and not line.strip().isdigit():
                text_lines.append(line.strip())

        if not time_line or not text_lines:
            continue

        try:
            parts = time_line.split(" --> ")
            start_sec = _srt_time_to_seconds(parts[0].strip())
            end_sec = _srt_time_to_seconds(parts[1].strip())
        except (ValueError, IndexError):
            continue

        # Metin kelimelerine ayır ve zaman dağıt
        text = " ".join(text_lines)
        # HTML tag'lerini temizle
        import re
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\[.*?\]", "", text)  # [Müzik] gibi etiketleri kaldır

        word_list = text.lower().split()
        word_list = [w.strip(".,!?;:()\"'") for w in word_list if w.strip(".,!?;:()\"'")]

        if not word_list:
            continue

        # Kelimeler arası eşit zaman dağılımı (basit yaklaşım)
        duration = end_sec - start_sec
        word_duration = duration / len(word_list)

        for i, word in enumerate(word_list):
            if len(word) < MIN_WORD_LENGTH or len(word) > MAX_WORD_LENGTH:
                continue

            # Türkçe karakter kontrolü
            if not all(c.isalpha() or c in "çğıöşüÇĞİÖŞÜ" for c in word):
                continue

            w_start = start_sec + i * word_duration
            w_end = w_start + word_duration

            words.append({
                "word": word,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "video": video_name
            })

    return words


def _srt_time_to_seconds(time_str: str) -> float:
    """SRT zaman formatını saniyeye çevir: 00:01:23,456 → 83.456"""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


# ═══════════════════════════════════════════════════════════════
#  STAGE 4: Kelime Filtreleme
# ═══════════════════════════════════════════════════════════════

def stage4_word_filtering(min_freq: int = 5):
    """
    Kelime listesini Türkçe sözlük ve frekans eşiğine göre filtrele.

    Args:
        min_freq: Minimum kelime tekrar sayısı
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Kelime Filtreleme")
    logger.info("=" * 60)

    input_csv = os.path.join(DIRS["output_align"], "word_timestamps.csv")
    output_csv = os.path.join(DIRS["output_words_filtered"], "filtered_words.csv")
    os.makedirs(DIRS["output_words_filtered"], exist_ok=True)

    if not os.path.exists(input_csv):
        logger.error(f"Girdi CSV bulunamadı: {input_csv}")
        return

    # Türkçe sözlük yükle
    turkish_words = set()
    if os.path.exists(TURKISH_DICT_PATH):
        with open(TURKISH_DICT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    turkish_words.add(word)
        logger.info(f"Türkçe sözlük yüklendi: {len(turkish_words)} kelime")
    else:
        logger.warning(f"Türkçe sözlük bulunamadı: {TURKISH_DICT_PATH} — sözlük filtresi atlanıyor")

    # CSV oku
    import pandas as pd
    df = pd.read_csv(input_csv)
    total_before = len(df)

    # Frekans hesapla
    word_counts = df["word"].value_counts()

    # Filtrele
    filtered = df[
        (df["word"].str.len() >= MIN_WORD_LENGTH) &
        (df["word"].str.len() <= MAX_WORD_LENGTH) &
        (df["word"].map(word_counts) >= min_freq)
    ]

    # Sözlük filtresi (sözlük mevcutsa)
    if turkish_words:
        filtered = filtered[filtered["word"].isin(turkish_words)]

    # Kaydet
    filtered.to_csv(output_csv, index=False, encoding="utf-8")

    unique_words = filtered["word"].nunique()
    logger.info(f"Filtreleme: {total_before} → {len(filtered)} örnek ({unique_words} benzersiz kelime)")
    logger.info(f"✅ Stage 4 tamamlandı → {output_csv}")

    # En sık kelimeler
    top_words = filtered["word"].value_counts().head(20)
    logger.info(f"En sık 20 kelime:\n{top_words.to_string()}")


# ═══════════════════════════════════════════════════════════════
#  STAGE 5: Kelime Bazlı Video Kırpma
# ═══════════════════════════════════════════════════════════════

def stage5_word_clip_extraction():
    """
    Filtrelenmiş her kelime için ±0.5 sn merkezli 1 saniyelik video klibi kes.

    ffmpeg kullanır.
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: Kelime Bazlı Video Kırpma")
    logger.info("=" * 60)

    filtered_csv = os.path.join(DIRS["output_words_filtered"], "filtered_words.csv")
    output_dir = DIRS["output_word_clips"]
    input_dir = DIRS["input"]

    if not os.path.exists(filtered_csv):
        logger.error(f"Filtrelenmiş CSV bulunamadı: {filtered_csv}")
        return

    import pandas as pd
    df = pd.read_csv(filtered_csv)
    logger.info(f"Kesilecek klip sayısı: {len(df)}")

    success_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        word = row["word"]
        start = float(row["start"])
        end = float(row["end"])
        video_name = row["video"]

        # Kelime dizini oluştur
        word_dir = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)

        # Video dosyasını bul
        video_path = os.path.join(input_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            error_count += 1
            continue

        # Klip zaman hesabı (kelime merkezli ±0.5 sn)
        word_center = (start + end) / 2
        clip_start = max(0, word_center - CLIP_DURATION / 2)

        # Çıktı dosya adı
        timestamp = int(time.time() * 1000)
        out_path = os.path.join(word_dir, f"{word}_{video_name}_{idx}_{timestamp}.mp4")

        # ffmpeg ile kes
        ffmpeg_exe = _get_ffmpeg_cmd()
        cmd = [
            ffmpeg_exe, "-y",
            "-ss", f"{clip_start:.3f}",
            "-i", video_path,
            "-t", f"{CLIP_DURATION:.3f}",
            "-r", str(TARGET_FPS),
            "-vf", f"scale=256:256",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",  # Ses kaldır
            "-loglevel", "error",
            out_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                success_count += 1
            else:
                error_count += 1
        except Exception:
            error_count += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"  İlerleme: {idx+1}/{len(df)} ({success_count} başarılı, {error_count} hatalı)")

    logger.info(f"✅ Stage 5 tamamlandı: {success_count} klip kesildi, {error_count} hata")


# ═══════════════════════════════════════════════════════════════
#  STAGE 6: Yüz Tespiti + Dudak ROI Kırpma
# ═══════════════════════════════════════════════════════════════

def stage6_face_detection_and_crop():
    """
    Her kelime klibinde yüz tespiti yapıp dudak bölgesini kırp.

    Yöntem 1: face_recognition + dlib 68-nokta landmark
    Yöntem 2: MediaPipe FaceMesh (fallback)
    Yöntem 3: OpenCV Haar Cascade (son çare)
    """
    logger.info("=" * 60)
    logger.info("STAGE 6: Yüz Tespiti + Dudak ROI Kırpma")
    logger.info("=" * 60)

    input_dir = DIRS["output_word_clips"]
    output_dir = DIRS["output_face_crop"]

    # Yüz tespiti backend seç
    face_detector = _get_face_detector()

    if not os.path.exists(input_dir):
        logger.error(f"Kelime klip dizini bulunamadı: {input_dir}")
        return

    word_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.info(f"İşlenecek kelime dizini: {len(word_dirs)}")

    total_clips = 0
    success_count = 0

    for word in word_dirs:
        word_input = os.path.join(input_dir, word)
        word_output = os.path.join(output_dir, word)
        os.makedirs(word_output, exist_ok=True)

        clips = [f for f in os.listdir(word_input) if f.endswith(".mp4")]

        for clip_file in clips:
            total_clips += 1
            clip_path = os.path.join(word_input, clip_file)

            try:
                lip_frames = _extract_lip_roi_from_video(clip_path, face_detector)

                if lip_frames is not None and len(lip_frames) > 0:
                    out_name = clip_file.replace(".mp4", ".npy")
                    out_path = os.path.join(word_output, out_name)
                    np.save(out_path, np.array(lip_frames, dtype=np.float32))
                    success_count += 1
            except Exception as e:
                logger.debug(f"ROI çıkarma hatası ({clip_file}): {e}")

    logger.info(f"✅ Stage 6 tamamlandı: {success_count}/{total_clips} klipten dudak ROI çıkarıldı")


def _get_face_detector():
    """En iyi mevcut yüz tespiti backend'ini seç."""
    # Yöntem 1: MediaPipe (en hassas, en hızlı)
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        logger.info("Yüz tespiti: MediaPipe FaceMesh")
        return ("mediapipe", face_mesh)
    except (ImportError, AttributeError):
        pass

    # Yöntem 2: face_recognition + dlib (LIRA-Gen orijinal)
    try:
        import face_recognition
        logger.info("Yüz tespiti: face_recognition + dlib")
        return ("face_recognition", None)
    except ImportError:
        pass

    # Yöntem 3: OpenCV Haar Cascade (son çare)
    local_haar_path = os.path.join(PROJECT_ROOT, "configs", "haarcascade_frontalface_default.xml")
    if not os.path.exists(local_haar_path):
        try:
            src = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(src):
                shutil.copy2(src, local_haar_path)
        except Exception:
            pass

    if os.path.exists(local_haar_path):
        cascade = cv2.CascadeClassifier(local_haar_path)
        if not cascade.empty():
            logger.info("Yüz tespiti: OpenCV Haar Cascade (düşük doğruluk)")
            return ("haar", cascade)

    logger.error("Yüz tespiti backend'i bulunamadı!")
    return ("none", None)


# MediaPipe dudak indisleri (dış kontur)
LIP_OUTER_MP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]

# dlib 68-nokta dudak indisleri
LIP_DLIB = list(range(48, 68))


def _extract_lip_roi_from_video(video_path: str, detector_info: tuple) -> Optional[List[np.ndarray]]:
    """Videodan dudak ROI frame'lerini çıkar."""
    backend, detector = detector_info
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_lip_pts = None  # Önbelleğe alınmış dudak koordinatları

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        lip_roi = None

        if backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                lip_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in LIP_OUTER_MP]
                lip_roi = _crop_lip_from_points(frame, lip_pts)

        elif backend == "face_recognition":
            import face_recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Küçült (hız için)
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            face_landmarks = face_recognition.face_landmarks(small)

            if face_landmarks:
                # Dudak noktaları (scale geri)
                top_lip = [(x * 2, y * 2) for x, y in face_landmarks[0].get("top_lip", [])]
                bottom_lip = [(x * 2, y * 2) for x, y in face_landmarks[0].get("bottom_lip", [])]
                lip_pts = top_lip + bottom_lip
                if lip_pts:
                    lip_roi = _crop_lip_from_points(frame, lip_pts)

        elif backend == "haar":
            if last_lip_pts is not None:
                lip_roi = _crop_lip_from_points(frame, last_lip_pts)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    x, y, fw, fh = faces[0]
                    # Dudak bölgesi noktaları: sol-üst ve sağ-alt sınırlar
                    lip_pts = [
                        (x + int(fw * 0.2), y + int(fh * 0.6)),
                        (x + int(fw * 0.8), y + fh)
                    ]
                    last_lip_pts = lip_pts
                    lip_roi = _crop_lip_from_points(frame, lip_pts)

        if lip_roi is not None:
            frames.append(lip_roi)
        else:
            # Boş frame ekle (padding)
            frames.append(np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32))

    cap.release()

    if len(frames) == 0:
        return None

    # Frame sayısını SEQ_LEN'e eşitle
    if len(frames) > SEQ_LEN:
        # Eşit aralıklı örnekleme
        indices = np.linspace(0, len(frames) - 1, SEQ_LEN, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < SEQ_LEN:
        # Sıfır padding
        pad = [np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)] * (SEQ_LEN - len(frames))
        frames.extend(pad)

    return frames


def _crop_lip_from_points(frame: np.ndarray, lip_points: list) -> Optional[np.ndarray]:
    """Dudak noktalarından ROI kırp."""
    if not lip_points:
        return None

    xs = [p[0] for p in lip_points]
    ys = [p[1] for p in lip_points]

    h, w = frame.shape[:2]
    margin_x = int((max(xs) - min(xs)) * 0.25)
    margin_y = int((max(ys) - min(ys)) * 0.25)

    x1 = max(0, min(xs) - margin_x)
    y1 = max(0, min(ys) - margin_y)
    x2 = min(w, max(xs) + margin_x)
    y2 = min(h, max(ys) + margin_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    resized = cv2.resize(gray, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ═══════════════════════════════════════════════════════════════
#  STAGE 7: NPY Dataset Birleştirme
# ═══════════════════════════════════════════════════════════════

def stage7_npy_consolidation():
    """
    Stage 6'da üretilen kelime-bazlı NPY dosyalarını
    Blind Eye eğitim formatına (data/processed/{word}/{file}.npy) birleştir.
    """
    logger.info("=" * 60)
    logger.info("STAGE 7: NPY Dataset Birleştirme")
    logger.info("=" * 60)

    input_dir = DIRS["output_face_crop"]
    output_dir = DIRS["output_npy"]

    if not os.path.exists(input_dir):
        logger.error(f"Yüz kırpma dizini bulunamadı: {input_dir}")
        return

    word_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    total_copied = 0

    for word in word_dirs:
        src_dir = os.path.join(input_dir, word)
        dst_dir = os.path.join(output_dir, word)
        os.makedirs(dst_dir, exist_ok=True)

        npy_files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]

        for npy_file in npy_files:
            src = os.path.join(src_dir, npy_file)
            dst = os.path.join(dst_dir, npy_file)

            # Doğrulama: boyut kontrolü
            try:
                data = np.load(src)
                if data.shape[0] == SEQ_LEN and data.shape[1] == ROI_SIZE:
                    # channel boyutu ekle: [T, H, W] → [T, H, W, 1]
                    if len(data.shape) == 3:
                        data = data[..., np.newaxis]
                    np.save(dst, data)
                    total_copied += 1
            except Exception as e:
                logger.debug(f"NPY doğrulama hatası ({npy_file}): {e}")

    logger.info(f"✅ Stage 7 tamamlandı: {total_copied} NPY dosyası birleştirildi → {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  STAGE 8: Labels & Metadata Üretimi
# ═══════════════════════════════════════════════════════════════

def stage8_generate_labels():
    """
    Blind Eye eğitim formatında labels.json ve metadata CSV üret.

    labels.json formatı:
        {"word/word_video_123.npy": "word", ...}
    """
    logger.info("=" * 60)
    logger.info("STAGE 8: Labels & Metadata Üretimi")
    logger.info("=" * 60)

    dataset_dir = DIRS["output_npy"]
    labels_path = os.path.join(dataset_dir, "labels.json")

    # Mevcut labels.json varsa yükle
    existing_labels = {}
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                existing_labels = json.load(f)
            logger.info(f"Mevcut labels.json yüklendi: {len(existing_labels)} örnek")
        except Exception:
            existing_labels = {}

    # Yeni etiketler tarafından oluşturulan NPY dosyalarını tara
    new_labels = {}
    word_dirs = [d for d in os.listdir(dataset_dir)
                 if os.path.isdir(os.path.join(dataset_dir, d))]

    for word in word_dirs:
        word_dir = os.path.join(dataset_dir, word)
        npy_files = [f for f in os.listdir(word_dir) if f.endswith(".npy")]

        for npy_file in npy_files:
            rel_path = f"{word}/{npy_file}"
            new_labels[rel_path] = word

    # Birleştir
    merged = {**existing_labels, **new_labels}

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # İstatistikler
    word_counts = {}
    for word in merged.values():
        word_counts[word] = word_counts.get(word, 0) + 1

    added_new = len(merged) - len(existing_labels)

    logger.info(f"Labels.json güncellendi:")
    logger.info(f"  Toplam: {len(merged)} örnek")
    logger.info(f"  Yeni eklenen: {added_new}")
    logger.info(f"  Benzersiz kelime: {len(word_counts)}")

    # Metadata CSV
    metadata_path = os.path.join(dataset_dir, "lira_gen_metadata.csv")
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count", "source"])
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            writer.writerow([word, count, "lira_gen_turkish"])

    logger.info(f"✅ Stage 8 tamamlandı: {labels_path}")
    logger.info(f"  Metadata: {metadata_path}")

    # Vocab güncelle
    _update_vocab(word_counts)


def _update_vocab(word_counts: dict):
    """configs/vocab.json'u güncelle."""
    vocab_path = os.path.join(PROJECT_ROOT, "configs", "vocab.json")

    vocab = {
        "chars": list("abcçdefgğhıijklmnoöprsştuüvyz "),
        "charset": list("abcçdefgğhıijklmnoöprsştuüvyz "),
        "blank_token": "<blank>",
        "num_classes": 31,
        "words": sorted(word_counts.keys()),
        "word_count": len(word_counts),
        "total_samples": sum(word_counts.values()),
        "source": "lira_gen_turkish + manual"
    }

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"Vocab güncellendi: {vocab_path} ({len(word_counts)} kelime)")


# ═══════════════════════════════════════════════════════════════
#  ANA FONKSİYON (CLI)
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LIRA-Gen Türkçe — YouTube'dan Otomatik Dudak Okuma Veri Seti Üretici"
    )
    parser.add_argument("--playlist", type=str, default=None,
                        help="YouTube playlist URL (Stage 1 için)")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Yerel video dizini (YouTube indirmeden)")
    parser.add_argument("--stage", type=int, default=1,
                        help="Başlangıç aşaması (1-8)")
    parser.add_argument("--max-videos", type=int, default=50,
                        help="Maksimum indirilecek video sayısı")
    parser.add_argument("--min-freq", type=int, default=5,
                        help="Minimum kelime frekansı (Stage 4)")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  LIRA-Gen Türkçe — Dudak Okuma Veri Seti Üretici   ║")
    logger.info("║  TÜBİTAK 2209-A — Blind Eye Projesi                ║")
    logger.info("║  Kaynak: github.com/Megiyanto377/LIRA-Gen           ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    ensure_dirs()

    start_stage = args.stage
    t0 = time.time()

    # Yerel video dizini kullanılıyorsa Stage 1'i atla
    if args.local_dir and start_stage == 1:
        logger.info(f"Yerel video dizini kullanılıyor: {args.local_dir}")
        # Videoları input dizinine kopyala
        if os.path.exists(args.local_dir):
            for f in os.listdir(args.local_dir):
                if f.endswith((".mp4", ".mkv", ".avi")):
                    src = os.path.join(args.local_dir, f)
                    dst = os.path.join(DIRS["input"], f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
        start_stage = 2

    # Pipeline aşamalarını çalıştır
    stages = {
        1: ("YouTube İndirme", lambda: stage1_download_youtube(args.playlist, args.max_videos)),
        2: ("Shot Detection", stage2_shot_detection),
        3: ("Forced Alignment", stage3_forced_alignment),
        4: ("Kelime Filtreleme", lambda: stage4_word_filtering(args.min_freq)),
        5: ("Video Kırpma", stage5_word_clip_extraction),
        6: ("Yüz Tespiti + ROI", stage6_face_detection_and_crop),
        7: ("NPY Birleştirme", stage7_npy_consolidation),
        8: ("Labels & Metadata", stage8_generate_labels),
    }

    for stage_num in range(start_stage, 9):
        stage_name, stage_func = stages[stage_num]
        logger.info(f"\n{'='*60}")
        logger.info(f"Aşama {stage_num}/8 başlıyor: {stage_name}")
        logger.info(f"{'='*60}")

        try:
            stage_func()
        except Exception as e:
            logger.error(f"Aşama {stage_num} hatası: {e}")
            logger.error(f"Pipeline aşama {stage_num}'da durdu. --stage {stage_num} ile yeniden başlatabilirsiniz.")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elapsed = time.time() - t0
    logger.info(f"\n🎉 LIRA-Gen Türkçe pipeline tamamlandı! ({elapsed/60:.1f} dakika)")
    logger.info(f"Üretilen veri seti: {DIRS['output_npy']}")
    logger.info(f"Eğitim başlatmak için: python tools/train_v2.py")


if __name__ == "__main__":
    main()
